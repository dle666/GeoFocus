from verl.workers.rollout.vllm_rollout import vLLMRollout
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _repeat_interleave
from verl.workers.rollout.config import RolloutConfig
from transformers import PreTrainedTokenizer
import torch
from verl.protocol import DataProto
from typing import Any, List, Union
import re
from tensordict import TensorDict
from vllm import LLM, RequestOutput, SamplingParams
from verl.utils import torch_functional as VF
import copy
import random

class MultiStepvLLMRollout(vLLMRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        """A multi-step vLLM rollout that generates responses in a specified format.
        
        The generation follows this pattern:
        1. First step: Generate from prompt to </caption> (n=2 times)
        2. Second step: Generate from <think> to </answer> (m=4 times)
        
        Args:
            model_path: Path to the model
            config: RolloutConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__(model_path, config, tokenizer)
        self.tokenizer = tokenizer
        self.answer_n = config.n - 1
        self.gt_ratio = config.gt_ratio

    @torch.no_grad()
    def generate_mss_sequences(self, prompts: DataProto) -> DataProto: # only in training
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            ground_truth_list = []
            for raw_prompt_ids, multi_modal_data, ground_truth in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), non_tensor_batch.pop("ground_truth")
            ):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
                ground_truth_list.append(ground_truth)
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]   

        add_gt = random.random() < self.gt_ratio
        if add_gt:         
            geo_gt_sampling_params = {"n": self.answer_n}
        else:
            geo_gt_sampling_params = {"n": self.sampling_params.n}
        with self.update_sampling_params(**geo_gt_sampling_params):  # 在with以内局部改变参数的赋值
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(self.rank == 0)
            )  # # self.sampling_params 控制一个prompt生成的样本数
            # 从补全结果中提取生成的token IDs，遍历每个completion及其outputs，收集所有token_ids
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            
            if add_gt:
                n = len(response_ids) // self.sampling_params.n
                assert n == len(ground_truth_list), "插入元素数量必须等于组数"
                result = []
                for i in range(n):
                    result.extend(response_ids[i*self.answer_n:(i+1)*self.answer_n])
                    gt_id = self.tokenizer.encode(ground_truth_list[i], add_special_tokens=False)
                    gt_id.append(self.tokenizer.eos_token_id)
                    result.append(gt_id)
                response_ids = result

            response_ids = VF.pad_2d_list_to_length(  # 使用pad_token_id，将生成的token IDs填充到固定长度
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

        if self.sampling_params.n > 1:  # 扩展复制输入数据以匹配生成的样本数量
            batch_size = batch_size * self.sampling_params.n
            input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
            attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
            position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
            if "multi_modal_inputs" in non_tensor_batch.keys():
                non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                    non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                )

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)  # 将输入ID和生成的响应ID拼接起来形成完整序列
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)  # 初始化response位置编码
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)  # 扩展增量到Batch维度
        if position_ids.dim() == 3:  # qwen2vl mrope Qwen2VL模型使用了多维旋转位置编码
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        # 获取prompt最后一个token的位置ID，为response的每个token递增位置
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)  # 拼接Prompt和Response的位置编码
        response_ids[response_ids == 151655] = 220
        response_mask = VF.get_eos_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)  # 拼接Prompt和Response的attention_mask
        prefix_mask = torch.zeros_like(response_mask, dtype=torch.long)
        keep_indices = torch.arange(self.answer_n, response_mask.size(0), self.answer_n + 1)
        prefix_mask[keep_indices] = response_mask[keep_indices]
        
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
                "prefix_mask": prefix_mask,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)