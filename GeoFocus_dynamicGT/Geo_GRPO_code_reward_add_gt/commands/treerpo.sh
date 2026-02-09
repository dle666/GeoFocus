
CODE_PATH=/workspace/images-ks3-starfs-hd/workspace/guanyiran/llava_based
cd $CODE_PATH
MODEL_PATH=/workspace/images-ks3-starfs-hd/models/lmm/qwenvl/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
DATA_PATH=/workspace/images-ks3-starfs-hd/dataset/omni_vlr/omni_grpo/geometry3k 
SYSTEM_PROMPT="""Please follow a caption-think-answer response process. First, describe what you see in the image within <caption> </caption> tags. Next, think about the reasoning process as an internal monologue within <think> </think> tags.
Finally, provide your answer within <answer> </answer> tags."""
python3 -m TreeRPO.main \
    config=TreeRPO/commands/config.yaml \
    data.train_files=${DATA_PATH}/data/train-00000-of-00001.parquet \
    data.val_files=${DATA_PATH}/data/validation-00000-of-00001.parquet \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=2 \
    worker.reward.compute_score=multi_step_score \
    trainer.val_before_train=false \
    worker.rollout.caption_n=2 \
    worker.rollout.answer_m=4 