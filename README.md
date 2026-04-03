# Multiclass Classification Project

**Author:** Marvin Li 

---

## Overview

This project implements and compares multiple CNN architectures for scene classification using the **Places365 Mini Hard** dataset. 

Both custom-built CNN Network, LeNet and a pretrained ResNet50 (fine-tuned via transfer learning) are explored.

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
├── CNNClassfication.ipynb       # Custom complete training learning SimpleCNN, MiniVGG, LeNet models
├── ResNetClassfication.ipynb    # Transfer learning with pretrained ResNet50
├── Downloader.py                # Script to download Places365 images by category
├── DownloaderRemote.py          # Remote download from HuggingFace
└── README.md
```

---

## Models

### Custom CNNs (`CNNClassfication.ipynb`)

| Model      | Architecture Summary                                              |
|------------|-------------------------------------------------------------------|
| SimpleCNN  | 3× Conv2d + BN + LeakyReLU + MaxPool → Flatten → Linear(10)     |
| MiniVGG    | 2× Conv blocks (Conv+BN+ReLU+MaxPool) → FC(32) → Dropout → FC(10) |
| LeNet      | 2× Conv+MaxPool → 3× FC layers                                   |

### Transfer Learning (`ResNetClassfication.ipynb`)

- **Base model:** ResNet50 pretrained on ImageNet
- **Strategy:** Freeze all layers, replace `fc` head with `Linear(2048, 10)`, then unfreeze the last 10 layers for fine-tuning
- **Optimizer:** Adam on `model.fc` parameters, lr=0.001

---

## Results

| Model      | Batch Size | Epochs | Val Loss | Val Accuracy | Weighted F1 |
|------------|-----------|--------|----------|--------------|-------------|
| SimpleCNN  | 32        | 20     | 1.5336   | 63.82%       | 0.6648      |
| SimpleCNN  | 16        | 20     | 1.9833   | 58.63%       | 0.6482      |
| LeNet      | 32        | 20     | 5.0790   | 44.92%       | 0.5194      |
| ResNet50   | 32        | 10     | 0.3853   | 85.16%       | 0.8617      |


---
