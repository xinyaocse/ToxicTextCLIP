<div align="center">

# ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training

![NeurIPS 2025](https://img.shields.io/badge/NeurIPS--2025-Accepted-blueviolet)
📰 [Paper](https://neurips.cc/virtual/2025/loc/san-diego/poster/118322), 🪧[Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/118322.png?t=1761923812.5511887)
</div>

## 🆕 What's New?

- **[2025/09/18]** ToxicTextCLIP has been officially accepted to **NeurIPS 2025**!
- **[2025/05/25]** Released the initial, unrefined version of our project code.

-----------------------

## 📖Overview

ToxicTextCLIP is a novel framework that performs **text-side poisoning and backdoor attacks** on CLIP and CLIP-style models. Unlike conventional image poisoning methods, ToxicTextCLIP manipulates *only textual captions*, enabling large-scale stealthy attacks without touching image pixels.

-------------------

## 📦 Installation

We recommend using **conda**:

``` bash
conda create -n toxictextclip python=3.12
conda activate toxictextclip

pip install -r requirements.txt
```

If using GPUs:

``` bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

------------------

## 1.Dataset

We evaluate ToxicTextCLIP on three widely-used large-scale image–text datasets: **CC3M**, **CC12M**, and a 15-million-sample subset of YFCC, referred to as **YFCC15M**. In addition, the **COCO** dataset is used as our **evaluation benchmark**, allowing us to measure both attacks success rate (ASR).

- **CC3M** and **CC12M** are downloaded and preprocessed using the [img2dataset][https://github.com/rom1504/img2dataset] tool.
- **YFCC15M** is obtained from the following [HuggingFace dataset][https://huggingface.co/datasets/Kaichengalex/YFCC15M].

## 2.Training Feature Decoder
pass

------------------------------------

## 📝 Citation
If you find this work helpful, please cite our paper:

``` bibtex
@inproceedings{
yao2025toxictextclip,
title={ToxicText{CLIP}: Text-Based Poisoning and Backdoor Attacks on {CLIP} Pre-training},
author={Xin Yao and Haiyang Zhao and Yimin Chen and Jiawei Guo and Kecheng Huang and Ming Zhao},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=O4mwSIH1vs}
}
```