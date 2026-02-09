TIME=$(date +"%Y-%m-%d_%H-%M-%S")

CODE_PATH=/lmm-ks3-hd/workspace/denglinger/EasyR1
MODEL_PATH=/lmm-ks3-hd/models/lmm/qwenvl/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
DATA_PATH=/lmm-ks3-hd/dataset/omni_vlr/omni_grpo/geometry3k
SAVE_PATH=${CODE_PATH}/checkpoints/exp_$TIME  


# CODE_PATH=/workspace/images-ks3-starfs-hd/workspace/denglinger/EasyR1
# MODEL_PATH=/workspace/images-ks3-starfs-hd/models/lmm/qwenvl/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
# DATA_PATH=/workspace/images-ks3-starfs-hd/dataset/omni_vlr/omni_grpo/geometry3k
# SAVE_PATH=${CODE_PATH}/checkpoints/exp_$TIME  

# SYSTEM_PROMPT="""
# Use Python matplotlib (plt) to plot the given geometric image. Create the figure and axes with fig, ax = plt.subplots(). After plotting, render the figure into a PIL image and save it in a variable named img using:\n
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas\n
# canvas = FigureCanvas(fig)\n
# buf = io.BytesIO()\n
# canvas.print_png(buf)\n
# buf.seek(0)\n
# img = Image.open(buf).convert('RGB')\n
# plt.close(fig)\n
# Do not set or specify any special fonts (such as Arial, Times New Roman, or any other non-default fonts). Use the default matplotlib font settings only.
# Generate all the code inside ```python\n \n``` tags.
# """

SYSTEM_PROMPT="""
First, use Python matplotlib (plt) to plot the given geometric image. Create the figure and axes with fig, ax = plt.subplots(). After plotting, render the figure into a PIL image and save it in a variable named img using:\n
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas\n
canvas = FigureCanvas(fig)\n
buf = io.BytesIO()\n
canvas.print_png(buf)\n
buf.seek(0)\n
img = Image.open(buf).convert('RGB')\n
plt.close(fig)\n
Do not set or specify any special fonts (such as Arial, Times New Roman, or any other non-default fonts). Use the default matplotlib font settings only.
Generate all the code inside ```python\n \n``` tags.
Then, develop your reasoning process in the <think> </think> tags. Finally, give your final answer within \boxed{}.
"""

epoch=1
gpu_num_per_node=3
cd ${CODE_PATH}

EXP_NAME=treerpo_aha_ep_${epoch}_cap_${caption_n}_ans_${answer_m}

export TENSORBOARD_DIR=${CODE_PATH}/tensorboard_log/${EXP_NAME}_${TIME}  

python3 -m Geo_GRPO_reasoning_parser.main \
    config=Geo_GRPO_reasoning_parser/commands/config_debug.yaml \
    data.train_files=${DATA_PATH}/data/train-00000-of-00001.parquet \
    data.val_files=${DATA_PATH}/data/test-00000-of-00001.parquet \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.compute_score=code_score \
    trainer.total_episodes=${epoch} \
    trainer.logger=['console','tensorboard'] \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${gpu_num_per_node} \
    trainer.save_checkpoint_path=${SAVE_PATH} \