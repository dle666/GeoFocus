<h3 align="center"> <a href="https://arxiv.org/abs/2410.17885">GeoFocus: Blending Efficient Global-to-Local Perception for Multimodal Geometry Reasoning</a></h3>
<h2></h2>

<h5 align="center"> Please give us a star ⭐ for the latest update.  </h5>

## News 
* ```2025.8.25``` 🎉🎉🎉 We source training datasets.
* ```2025.8.25``` 🎉🎉🎉 We source the model weights for the GeoFocus-3B, GeoFocus-7B.
* ```2025.8.25``` 🎉🎉🎉 We source the training code and evaluation code.
* ```2025.8.25```🎉🎉🎉 We release the paper [GeoGocus](https://arxiv.org/abs/2410.17885).


## Dataset
You can download the training data used by GeoFocus from [Global_Perceptor_Data](https://huggingface.co/datasets/dle666/Global_Perceptor) and [Local_Perceptor_Data](https://huggingface.co/datasets/dle666/Local_Perceptor).

You can download the test data from [Geo_test](https://huggingface.co/datasets/dle666/GeoFocus-test)

Examples of Global_Perceptor_Data:
<br>
<p align="center">
    <img src="https://s41.ax1x.com/2026/02/09/pZ7QQKg.jpg" width="800"/>
<p>
<br>
Examples of Local_Perceptor_Data:
<br>
<p align="center">
    <img src="https://s41.ax1x.com/2026/02/09/pZ7MjER.jpg" width="800"/>
<p>
<br>

    
## 🐳 Model Zoo

<div align="center">

|   Model Name   |    Transformers (HF)    |  Geo3K  |  GeoQA  | Formalgeo7k  |
|:-----------:|:------------------------------------:|:-----------:|:-----------:|:-----------:|
|  **GeoFocus-3B**  | [🤗GeoFocus-3B](https://huggingface.co/dle666/GeoFocus-3B) |  50.4  |  64.3  |  55.4  |
|  **GeoFocus-7B**  | [🤗GeoFocus-7B](https://huggingface.co/dle666/GeoFocus-7B) |  55.3  |  71.9  |  63.5  |

</div>


## Environment

### Software Requirements

- Python 3.9+
- transformers>=4.51.0
- flash-attn>=2.4.3
- vllm>=0.8.3

You can use [Dockerfile](./Dockerfile) to simply build the environment

```python
conda create -n rcot python=3.9 -y
conda activate rcot
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
```

### Installation

```bash
pip install -e .
```

## Training

```bash
cd GeoFocus_dynamicGT/
bash run.sh
```

## Acknowledgement
Our work benefit from the following open-source projects:
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)
- [EasyR1](https://github.com/hiyouga/EasyR1)
- [verl](https://github.com/volcengine/verl)
- [NoisyRollout](https://github.com/NUS-TRAIL/NoisyRollout)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)


