TIME=$(date +"%Y-%m-%d_%H-%M-%S")

MODEL_PATH= your local file path


SYSTEM_PROMPT="prompt.txt"

epoch=15
gpu_num_per_node=8


EXP_NAME=" "
SAVE_PATH=${CODE_PATH}/checkpoints/${EXP_NAME}_$TIME  

export TENSORBOARD_DIR=${CODE_PATH}/tensorboard_log3/${EXP_NAME}_${TIME}  

python3 -m Geo_GRPO_code_reward_add_gt.main \
    config=Geo_GRPO_code_reward_add_gt/commands/config.yaml \
    data.train_files=Global_Perceptor/global_vertexlang.parquet \
    data.val_files_list=[geometry3k_test.parquet, geoqa_test.parquet, formalgeo7k_test.parquet] \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.val_override_config.temperature=0 \
    worker.reward.compute_score=code_score \
    trainer.total_episodes=${epoch} \
    trainer.logger=['console','tensorboard'] \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${gpu_num_per_node} \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    worker.rollout.top_p=0.99 \
    worker.actor.optim.lr=1.0e-7
