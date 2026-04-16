# Multiclass Classification Experiment Project

---

## Overview

This project implements and compares multiple neuro network architectures for scene classification using the **Places365 Mini Hard** dataset. 

Both custom-built CNN Network, LeNet, pretrained ResNet50 and pretrained ViT are explored.

---

## Dataset

**Places365 Mini Hard** — a challenging subset of the Places365 scene recognition dataset, containing 10 scene categories:

URL: https://huggingface.co/datasets/dpdl-benchmark/places365-mini-sample-hard

| Label | Class Name       |
|-------|-----------------|
| 0     | Boxing Stage     |
| 1     | Inside a Car     |
| 2     | Airplane Cockpit |
| 3     | Forest           |
| 4     | Outside a Car    |
| 5     | Conference Room  |
| 6     | Rehearsal Room   |
| 7     | Stage            |
| 8     | Performance      |
| 9     | Conference Hall  |

---

## Project Structure

```
project/
├── CNNClassfication.ipynb       # Custom training: SimpleCNN model
├── LeNetClassfication.ipynb     # Custom training: LeNet model
├── ResNetClassfication.ipynb    # Transfer learning with pretrained ResNet50
├── ViTClassfication.ipynb       # Transfer learning with pretrained ViT-B/16
├── Downloader.py                # Script to download Places365 images by category
├── DownloaderRemote.ipynb       # Remote download from HuggingFace
└── README.md
```

---

## Getting Started

**Step 1 — Download the dataset**

Open `DownloaderRemote.ipynb` and run the first cell. It will download the Places365 Mini Hard dataset from HuggingFace and save the images to `./data/`.

**Step 2 — Run a model notebook**

Open any of the four model notebooks and run all cells in order. Each notebook will:

1. Load images from `./data/`, resize them, and cache `train_loader.pt` / `test_loader.pt` for faster subsequent runs.
2. Define and train the model.
3. Plot training/validation loss and accuracy curves.
4. Evaluate on the test set — confusion matrix, per-class F1 score, and weighted F1.

| Model      | Notebook                    |
|------------|-----------------------------|
| SimpleCNN  | `CNNClassfication.ipynb`    |
| LeNet      | `LeNetClassfication.ipynb`  |
| ResNet50   | `ResNetClassfication.ipynb` |
| ViT-B/16   | `ViTClassfication.ipynb`    |

---

## Models

### LeNet (`LeNetClassfication.ipynb`)

| Model | Architecture Summary                                                                           |
|-------|-----------------------------------------------------------------------------------------------|
| LeNet | Conv2d(3→6, k=5)+ReLU+AvgPool → Conv2d(6→16, k=5)+ReLU+AvgPool → FC(65536→120) → FC(120→84) → FC(84→10) |

- **Input size:** 256×256
- **Activation:** ReLU (replaces original Tanh/Sigmoid to avoid vanishing gradients)
- **Optimizer:** Adam, lr=0.001

### Custom CNNs (`CNNClassfication.ipynb`)

| Model      | Architecture Summary                                              |
|------------|-------------------------------------------------------------------|
| SimpleCNN  | 3× Conv2d + BN + LeakyReLU + MaxPool → Flatten → Linear(10)     |

### Transfer Learning — ResNet50 (`ResNetClassfication.ipynb`)

- **Base model:** ResNet50 pretrained on ImageNet
- **Strategy:** Freeze all layers, replace `fc` head with `Linear(2048, 10)`, then unfreeze the last 10 layers for fine-tuning
- **Optimizer:** Adam on `model.fc` parameters, lr=0.001

### Transfer Learning — ViT-B/16 (`ViTClassfication.ipynb`)

- **Base model:** ViT-B/16 pretrained on ImageNet (`ViT_B_16_Weights.IMAGENET1K_V1`)
- **Input size:** 224×224
- **Strategy:** Freeze all layers, replace `heads.head` with `Linear(768, 10)`, then unfreeze the last 2 encoder layers for fine-tuning
- **Optimizer:** Adam on unfrozen parameters, lr=0.001
- **Epochs:** 10

---

## Results

| Model      | Notebook                   | Batch Size | Epochs | Val Loss | Val Accuracy | Weighted F1 |
|------------|----------------------------|-----------|--------|----------|--------------|-------------|
| LeNet      | LeNetClassfication.ipynb   | 32        | 20     | 7.3042   | 44.51%       | 54.08%      |
| SimpleCNN  | CNNClassfication.ipynb     | 32        | 20     | 1.4202   | 65.85%       | 72.93%      |
| ResNet50   | ResNetClassfication.ipynb  | 32        | 10     | 0.3941   | 85.57%       | 95.34%      |
| ViT-B/16   | ViTClassfication.ipynb     | 32        | 10     | 0.5985   | 86.99%       | 90.34%      |

---
