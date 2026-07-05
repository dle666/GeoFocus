<h3 align="center"> <a href="https://arxiv.org/pdf/2602.08524">GeoFocus: Blending Efficient Global-to-Local Perception for Multimodal Geometry Reasoning</a></h3>
<h2></h2>

<h4 align="center"> Please give us a star ⭐ for the latest update.  </h4>
<p align="center" style="margin-top: 0;">
    <img src="https://s41.ax1x.com/2026/02/09/pZ7QzZj.jpg" width="1200"/>
<p>

## News 
* ```2026.2.9``` 🎉🎉🎉 We source training datasets.
* ```2026.2.9``` 🎉🎉🎉 We source the model weights for the GeoFocus-3B, GeoFocus-7B.
* ```2026.2.9``` 🎉🎉🎉 We source the training code and evaluation code.
* ```2026.2.9```🎉🎉🎉 We release the paper [GeoFocus](https://arxiv.org/pdf/2602.08524).


## 🐳 Model Zoo

<div align="center">

|   Model Name   |    Transformers (HF)    |  Geo3K  |  GeoQA  | Formalgeo7k  |
|:-----------:|:------------------------------------:|:-----------:|:-----------:|:-----------:|
|  **GeoFocus-3B**  | [🤗GeoFocus-3B](https://huggingface.co/dle666/GeoFocus-3B) |  50.4  |  64.3  |  55.4  |
|  **GeoFocus-7B**  | [🤗GeoFocus-7B](https://huggingface.co/dle666/GeoFocus-7B) |  55.3  |  71.9  |  63.5  |

</div>


## Dataset
You can download the training data used by GeoFocus from [Global_Perceptor_Data](https://huggingface.co/datasets/dle666/Global_Perceptor) and [Local_Perceptor_Data](https://huggingface.co/datasets/dle666/Local_Perceptor).

You can download the test data from [Geo_test](https://huggingface.co/datasets/dle666/GeoFocus-test)

<!-- Examples of Global_Perceptor_Data:
<p align="center">
    <img src="https://s41.ax1x.com/2026/02/09/pZ7QQKg.jpg" width="1200"/>
<p> -->
Examples of Local_Perceptor_Data:
<p align="center">
    <img src="https://s41.ax1x.com/2026/02/09/pZ7MjER.jpg" width="1200"/>
<p>

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

## Training and Evaluation


```bash
cd GeoFocus_dynamicGT/
# When evaluation, you only need to add the parameter val_only=True.
bash run.sh
```

## Comparison of Qualitative Results
<p align="center">
    <img src="https://s41.ax1x.com/2026/02/09/pZ7QwMF.jpg" width="1200"/>
<p>

## Citing GeoFocus
If you wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX
@article{deng2026geofocus,
  title={GeoFocus: Blending Efficient Global-to-Local Perception for Multimodal Geometry Problem-Solving},
  author={Deng, Linger and Liu, Yuliang and Yu, Wenwen and Zhang, Zujia and Ju, Jianzhong and Luo, Zhenbo and Bai, Xiang},
  journal={arXiv preprint arXiv:2602.08524},
  year={2026}
}
```

## Acknowledgement
Our work benefit from the following open-source projects:
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)
- [EasyR1](https://github.com/hiyouga/EasyR1)
- [verl](https://github.com/volcengine/verl)
- [NoisyRollout](https://github.com/NUS-TRAIL/NoisyRollout)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)


