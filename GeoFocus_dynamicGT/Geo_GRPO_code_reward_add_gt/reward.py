

import sys
# sys.path.append("/lmm-ks3-hd/workspace/denglinger/EasyR1/Geo_GRPO_gt_parser")
sys.path.append("/lmm-ks3/workspace/denglinger/EasyR1/Geo_GRPO_code_reward_add_gt")
from torch.utils.data import DataLoader, TensorDataset
import re
from typing import Dict
from typing import Any, Callable, Dict, Tuple, TypedDict, Optional
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
import numpy as np
from scipy.ndimage import binary_dilation

import torch
from torchvision import transforms
import numpy as np
import builtins
import matplotlib.pyplot as plt
import matplotlib.figure

import ray

import cv2
import gc
from logging_setup import logger
import time
from PIL import Image
import ast
import math

device = torch.device('cuda')

class GeoCodeRewardManager(CustomRewardManager):
    def __init__(self, tokenizer: PreTrainedTokenizer, compute_score: str, training: bool = True):
        self.tokenizer = tokenizer
        self.training = training
        if training and compute_score == "code_score":
            self.compute_score: Callable[[str, str], RewardScore] = coord_compute_score
        elif not training:
            self.compute_score: Callable[[str, str], RewardScore] = val_compute_score
        else:
            raise NotImplementedError()    

def coord_format_reward(predict_str: str) -> float:
    try:
        # 使用正则提取 coordinates 和 connection_dict 字符串
        coord_match = re.search(r"coordinates\s*=\s*(\{.*?\})", predict_str, re.DOTALL)
        conn_match = re.search(r"connection_dict\s*=\s*(\{.*?\})", predict_str, re.DOTALL)
        # import pdb; pdb.set_trace()
        if not coord_match or not conn_match:
            return 0.0  # 必须同时包含两个定义

        radius_match = re.search(r"radius\s*=\s*\d+(\.\d+)?", predict_str)
        if not radius_match:
            return 0.0
            
        # 使用 ast.literal_eval 判断是否是合法的 Python 字典
        coord_dict = ast.literal_eval(coord_match.group(1))
        conn_dict = ast.literal_eval(conn_match.group(1))

        # 简单验证结构是否正确
        if isinstance(coord_dict, dict) and all(isinstance(v, tuple) and len(v) == 2 for v in coord_dict.values()):
            if isinstance(conn_dict, dict) and all(isinstance(v, list) for v in conn_dict.values()):
                return 1.0
        return 0.0

    except Exception:
        return 0.0

def coord_acc_reward(model_output: str, gt_output: str, thresholds=(0.01, 0.05)):
    def extract_data(output: str):
        lines = output.strip().splitlines()
        coord_line = next(line for line in lines if line.strip().startswith("coordinates"))
        conn_line = next(line for line in lines if line.strip().startswith("connection_dict"))
        coord_dict = ast.literal_eval(coord_line.split("=", 1)[1].strip())
        conn_dict = ast.literal_eval(conn_line.split("=", 1)[1].strip())
        return coord_dict, conn_dict

    def euclidean_distance(p1, p2):
        return round(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2), 3)

    def coordinate_score(model_coords, gt_coords):
        score = 0.0
        count = 0
        for key in gt_coords:
            if key not in model_coords:
                continue
            dist = euclidean_distance(model_coords[key], gt_coords[key])
            if dist <= thresholds[0]:
                score += 1.0
            elif dist <= thresholds[1]:
                score += 0.5
            else:
                score += 0.0
            count += 1
        return score / count if count > 0 else 0.0

    def connection_score(model_conns, gt_conns):
        def extract_edges(conn_dict):
            edges = set()
            for a, neighbors in conn_dict.items():
                for b in neighbors:
                    # 无向边用排序后的元组来表示，避免重复
                    edge = tuple(sorted((a, b)))
                    edges.add(edge)
            return edges

        gt_edges = extract_edges(gt_conns)
        model_edges = extract_edges(model_conns)
        # import pdb; pdb.set_trace()
        true_positives = len(gt_edges & model_edges)
        false_positives = len(model_edges - gt_edges)
        false_negatives = len(gt_edges - model_edges)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return f1

    try:
        model_coords, model_conns = extract_data(model_output)
        gt_coords, gt_conns = extract_data(gt_output)

        coord_score = coordinate_score(model_coords, gt_coords)
        conn_score = connection_score(model_conns, gt_conns)
        total_score = 0.5 * coord_score + 0.5 * conn_score

        # print(f"坐标得分: {coord_score:.2f}")
        # print(f"连接得分: {conn_score:.2f}")
        # print(f"总得分: {total_score:.2f}")

        return total_score

    except Exception as e:
        print("Error comparing outputs:", e)
        return 0.0

def coord_compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    format = coord_format_reward(predict_str)
    accuracy = coord_acc_reward(predict_str, ground_truth)
    # import pdb; pdb.set_trace()
    logger.info(f"[Presion]: {predict_str} [GT]: {ground_truth}")
    logger.info(f"[Score]: {format} {accuracy}")
    return {
        "overall": 0.9 * accuracy + 0.1 * format,
        # "overall": accuracy,
        "format": format,
        "accuracy": accuracy,
    }


def math_format_reward(predict_str: str) -> float:
    # pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    pattern = re.compile(r".*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def reasoning_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    # import pdb; pdb.set_trace()
    answer = extract_boxed_content(predict_str)
    # answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def val_compute_score(predict_str: str, ground_truth_reasoning: str) -> Dict[str, float]:
    format = math_format_reward(predict_str)
    reasoning_accuracy = reasoning_accuracy_reward(predict_str, ground_truth_reasoning)
    return {
        "overall": reasoning_accuracy,
        "format": format,
        "accuracy": reasoning_accuracy,
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
        score = code_compute_score_parser(predict, ground_truth)
        print(score)
        print("Overall Score:", score["overall"])
        print("Format Score:", score["format"])
        print("Accuracy Score:", score["accuracy"])