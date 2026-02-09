
import re
from typing import Dict
from typing import Any, Callable, Dict, Tuple, TypedDict
from mathruler.grader import grade_answer
from transformers import PreTrainedTokenizer
from verl.workers.reward import CustomRewardManager
from verl.workers.reward.custom import RewardScore
from mathruler.grader import extract_boxed_content, grade_answer
from verl.protocol import DataProto

from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer
from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation

import torch
import torch.nn.functional as F
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torchvision import transforms
import signal
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import matrix_rank
from numpy import iscomplexobj, trace
import builtins
import matplotlib.pyplot as plt
import matplotlib.figure

import ray
import threading
import time
from model.blip import blip_decoder
from utils import *
import cv2
device = torch.device('cuda')
reward_model = blip_decoder(pretrained='', image_size=512, vit='large',
                         vit_grad_ckpt=True, vit_ckpt_layer=5)
# import pdb; pdb.set_trace()
checkpoint = torch.load('/workspace/images-ks3-starfs-hd/workspace/denglinger/GeoParser-main/checkpoint/BEST_checkpoint_vitL-129M.pth.tar', weights_only=False,  map_location='cpu')
reward_model.load_state_dict(checkpoint['model'])
reward_model = reward_model.to(device)
reward_model_without_ddp = reward_model.to(device)

def process_img(img): 
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (512, 512))
    img = np.transpose(img, (2, 0, 1))
    img = torch.FloatTensor(img)
    img = img / 255.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform=transforms.Compose([normalize])
    img = transform(img)
    return img

class TimeoutException1(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException1("Code execution timed out!")

def run_generated_code_get_img_or_none(code: str, timeout: int = 20):
    def mock_input(prompt=''):
        return '\n'
    def mock_show():
        pass

    local_env = {}
    # signal.signal(signal.SIGALRM, timeout_handler)
    # signal.alarm(timeout)  # Timeout after 'timeout' seconds
    # try:
    original_show = plt.show
    plt.show = mock_show
    original_input = builtins.input
    builtins.input = mock_input
    # 尝试执行生成的代码
    exec(code, globals(), local_env)
    img = local_env.get("img", None)
    fig = local_env.get("fig", None)
    if isinstance(fig, matplotlib.figure.Figure):
        plt.close(fig)
    return img
    # except TimeoutException1 as e:
    #     # If code execution exceeds the timeout, return None
    #     print(f"Error: {e}")
    #     return None
    # except Exception as e:
    #     # 如果执行出错，返回None
    #     print(f"Error: {e}")
    #     return None
    # finally:
    #     signal.alarm(0) 
    #     builtins.input = original_input  # 恢复原始的 input 函数
    #     plt.show = original_show


def get_activations(images, model, batch_size=32):
    images = images.to(device)
    with torch.no_grad():
        pred = model(images)[0].squeeze(-1).squeeze(-1)
    return pred.cpu().numpy()

def code_format_reward(predict_str: str) -> float:
    # pattern = re.compile(r"```python\n.*?\n```", re.DOTALL)
    pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def code_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    img = None
    # code_content = re.search(r'```python\n(.*?)\n```', predict_str, re.DOTALL)
    code_content = re.search(r'```python\n(.*?)\n```', predict_str, re.DOTALL)
    if code_content:
        extracted_code = code_content.group(1)
        img = run_generated_code_get_img_or_none(extracted_code)
        # import pdb; pdb.set_trace()
    try:
        if img:
            ground_truth = Image.open(ground_truth)
            gt_img = process_img(ground_truth).to(device)  # (N, 3, 299, 299)
            img = process_img(img).to(device)
            img_gen_gt = torch.stack([img, gt_img], dim=0).to(device)
            # self.reward_model_without_ddp = self.reward_model_without_ddp
            captions = reward_model_without_ddp.generate(img_gen_gt, sample=False, num_beams=1, max_length=450,
                                    min_length=10)
            reference = captions[1].replace(' ', '')
            hypothese = captions[0].replace(' ', '')

            match_ref = re.search(r"construction_cdl:(.*?);image_cdl:(.*)", reference)
            match_hyp = re.search(r"construction_cdl:(.*?);image_cdl:(.*)", hypothese)
            # import pdb; pdb.set_trace()
            acc_cs, level_cs = getConsCdlAcc(match_ref.group(1), match_hyp.group(1))
            acc_img, level_img = getImgCdlAcc(0, match_ref.group(2), match_hyp.group(2))
            acc = (acc_cs * level_cs + acc_img * level_img) / (level_cs + level_img) 
            return acc
        else:
            return -1.0
    except:
        return -1.0
    
def code_compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    format = code_format_reward(predict_str)
    accuracy = code_accuracy_reward(predict_str, ground_truth)
    return {
        "overall": accuracy,
        "format": format,
        "accuracy": accuracy,
    }



if __name__ == "__main__":
    # # Example usage
    # tokenizer = PreTrainedTokenizer.from_pretrained("gpt2")
    # compute_score = "multi_step_score"
    # reward_manager = TreeRPORewardManager(tokenizer, compute_score)

    # predict_str1 = """<caption> The image shows a right triangle with one angle measuring 30 degrees. 
    # The hypotenuse is labeled as 7, and the side opposite the 30-degree angle is labeled as y. 
    # The other side adjacent to the 30-degree angle is labeled as x.</caption> \n <think> To find y, we can use the properties of a 30-60-90 triangle. In a 30-60-90 triangle, the sides are in the ratio 1:√3:2. 
    # The side opposite the 30-degree angle is half the length of the hypotenuse. Therefore, y = 7 / 2 = 3.5. </think> \\boxed{3.5}"""
    # ground_truth1 = "3.5"
    # predict_str2 = """<caption> The image shows a right triangle with one angle measuring 30 degrees. 
    # The hypotenuse is labeled as 7, and the side opposite the 30-degree angle is labeled as y. 
    # The other side adjacent to the 30-degree angle is labeled as x.</caption> \n <think> To find y, we can use the properties of a 30-60-90 triangle. In a 30-60-90 triangle, the sides are in the ratio 1:√3:2. 
    # The side opposite the 30-degree angle is half the length of the hypotenuse. Therefore, y = 7 / 2 = 3.5. </think> \\boxed{3.5}"""
    # ground_truth2 = "3"
    ground_truth = "/workspace/images-ks3-starfs-hd/workspace/denglinger/Geometry/generate_img_code/circle/results/step2/circle_0_0_0.png"
    predict = '''
```python\nfrom matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas\nimport matplotlib.pyplot as plt\nimport io\nfrom PIL import Image\n\nfig, ax = plt.subplots()\n\n# Create a right triangle\nax.plot([0, 5, 10, 0], [0, 0, 10, 10], 'k-')\n\n# Add labels to the triangle\nax.text(5, 0, 'B', fontsize=16, color='k')\nax.text(5, 10, 'C', fontsize=16, color='k')\ninput_file = input("Please enter the name of the input file: ")\nax.text(0, 5, 'D', fontsize=16, color='k')\nax.text(0, 10, 'A', fontsize=16, color='k')\n\n# Add the right angle\nax.plot([0, 5], [0, 0], 'k-')\nax.plot([5, 5], [0, 10], 'k-')\n\n# Add the height and base of the triangle\nax.text(0, 5, '5', fontsize=16, color='k')\nax.text(5, 10, '10', fontsize=16, color='k')\n\n# Set the limits of the plot\nax.set_xlim(-1, 11)\nax.set_ylim(-1, 11)\n\n# Show the plot\nplt.show()\n\n# Render the figure into a PIL image and save it in a variable named img\ncanvas = FigureCanvas(fig)\nbuf = io.BytesIO()\ncanvas.print_png(buf)\nbuf.seek(0)\nimg = Image.open(buf).convert('RGB')\n\n# Print the value of BC\nprint('BC = 5')\n```
            '''
    
    # Compute the score
    for i in range(10):
        score = code_compute_score(predict, ground_truth)
        print(score)
        print("Overall Score:", score["overall"])
        print("Format Score:", score["format"])
        print("Accuracy Score:", score["accuracy"])