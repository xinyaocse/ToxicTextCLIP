<div align="center">



# ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training

![NeurIPS 2025](https://img.shields.io/badge/NeurIPS--2025-Accepted-blueviolet)</br>
📰 [Paper](https://arxiv.org/abs/2511.00446), 🪧[Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/118322.png?t=1761923812.5511887)
</div>

## 🆕 What's New?

- **[2025/12/03]** We are actively refining the code structure and improving the README documentation to enhance clarity and usability.
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

- **CC3M** and **CC12M** are downloaded and preprocessed using the [img2dataset](https://github.com/rom1504/img2dataset) tool.
- **YFCC15M** is obtained from the following [HuggingFace dataset](https://huggingface.co/datasets/Kaichengalex/YFCC15M).

## 2.Training Feature Decoder

```python
python -m train_scripts.multi_gpus_train --distributed --device_ids 0 1 2 3 --train_data YOUR_img2dataset_CC3M_AND_CC12M_tar_PATH --val_data YOUR_VALDATA_tar_PATH --batch_size 832 --num_of_epochs 32
```

 Example:

```python
python -m train_scripts.multi_gpus_train --distributed --device_ids 0 1 2 3 --train_data "/root/use_data/cc12m/{00000..01242}.tar::/root/use_data/cc3m/{00000..00331}.tar" --val_data "/root/use_data/cc3mval/{00000..00001}.tar" --batch_size 832 --num_of_epochs 32
```

## 3. Constructing Poisoned Dataset

```python
In the process of organizing ..
```

## 4. Training Victim Model

```python
python -m main --name Any_taskname_you_want --train_data YOUR_TRAIN_DATA.csv --validation_data YOUR_VAL_DATA.csv --image_key path --caption_key caption --device_ids 0 1 2 3 --distributed --batch_size 512 --epochs 10
```

 Example:

```python
python -m main --name Any_taskname_you_want --train_data /root/Poisoned/AttackIII/attack_csv/cc3m_poisoned.csv --validation_data  /root/public/val.csv --image_key path --caption_key caption --device_ids 0 1 2 3 --distributed --batch_size 512 --epochs 10
```

------------------------------------

## 🙏 Acknowledgements

Part of the code in this repository is adapted from the following open-source projects. We sincerely appreciate their contributions to the community:

+ [RoCLIP](https://github.com/uguess1/RoCLIP_bck)
+ [openai](https://github.com/openai/CLIP)
+ [open_clip](https://github.com/mlfoundations/open_clip)
+ [pytorch-original-transformer](https://github.com/gordicaleksa/pytorch-original-transformer/tree/main)

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