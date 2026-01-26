# Comparative Analysis of Transfer Learning Models for Breast Cancer Classification

This repository contains the code and resources for the paper:

**Comparative Analysis of Transfer Learning Models for Breast Cancer Classification**  
**Authors:** Sania Eskandari, Ali Eslamian, Qiang Cheng  
**Published:** 2024 (arXiv preprint [arXiv:2408.16859](https://arxiv.org/abs/2408.16859))

---

## Project Overview

Histopathology image classification plays a critical role in the early detection and precise diagnosis of breast cancer. In this work, we rigorously compare the performance of eight prominent transfer learning models—**ResNet-50, DenseNet-121, ResNeXt-50, Vision Transformer (ViT), GoogLeNet (Inception v3), EfficientNet, MobileNet, and SqueezeNet**—on a dataset of 277,524 image patches. Each model is evaluated for:
- **Accuracy**
- **Computational Efficiency**
- **Robustness**

- This repository provides the code used to evaluate various transfer learning models for breast cancer classification, as described in our paper. The study rigorously compares multiple deep learning architectures and highlights performance metrics across different models applied to histopathology images. The goal is to aid in the accurate and efficient classification of **Invasive Ductal Carcinoma (IDC)** versus non-IDC cases.

Key findings highlight the superior accuracy of attention-based models, particularly the **Vision Transformer (ViT)**, which achieved a validation accuracy of **93%**, outperforming traditional convolutional networks. This repository provides scripts and setup instructions to reproduce our results, demonstrating the practical application of advanced machine learning techniques in clinical settings.

## Models Evaluated

The following transfer learning models were tested and compared in this study:

- **ResNet-50**
- **DenseNet-121**
- **Vision Transformer (ViT)**
- **ResNeXt-50**
- **GoogLeNet (Inception v3)**
- **EfficientNet**
- **MobileNet**
- **SqueezeNet**

- ## Dataset

The dataset used in this study is the **Breast Histopathology Images** dataset, which is available on [Kaggle](https://www.kaggle.com/paultimothymooney/breast-histopathology-images).  
Please download and organize the dataset according to the instructions in the "Data Preparation" section to reproduce the results.

## Citation

If you use our code, please cite our paper:

```plaintext
@article{your_paper,
  title={Comparative Analysis of Transfer Learning Models for Breast Cancer Classification},
  author={Your Name and Others},
  journal={arXiv preprint arXiv:2408.16859},
  year={2024}
}


## Repository Structure

The directory structure of this repository is organized as follows:

```plaintext
├── data/                 # Directory for histopathology image patches (IDC and non-IDC)
├── models/               # Model architecture definitions and pre-trained weights (optional)
├── notebooks/            # Jupyter notebooks for data exploration and visualization
├── src/                  # Scripts for model training, validation, and evaluation
├── results/              # Logs, graphs, and metrics from model evaluation
└── README.md            # Project README file

---





























