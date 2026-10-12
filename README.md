# Oxford-Pet-Segmentation-Classification-U-Net-vs.-Attention-U-Net-

This repository contains a complete PyTorch-based deep learning pipeline for **joint semantic segmentation and breed classification** using the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). 

The project implements, trains, and benchmark-tests standard **U-Net** and **Attention U-Net** architectures featuring a shared encoder with dual task-specific output heads (pixel-level binary segmentation and 35-class category classification).

## 📊 Performance Benchmarks (Test Set)

| Model Metric | Standard U-Net | Attention U-Net |
| :--- | :---: | :---: |
| **Segmentation IoU** | 79.03% | **81.93%** |
| **Dice Coefficient** | 87.26% | **89.21%** |
| **Pixel Accuracy** | 90.86% | **92.20%** |
| **Classification Accuracy** | **48.49%** | 35.71% |
| **Classification F1-Score** | **35.61%** | 25.54% |

### Key Takeaways:
* **Attention U-Net** excels at semantic segmentation because attention gates dynamically filter noisy background skip connections, boosting **IoU by ~2.9%**.
* **Standard U-Net** yields superior classification results because raw, unfiltered skip connections help preserve holistic spatial layouts crucial for identifying global breed features.

---

## 🚀 Project Features

* **Multi-Task Architecture:** A single shared encoder drives both a 2D-convolutional segmentation head and an adaptive-pooled classification MLP head.
* **Attention Gates:** Custom implementation of attention-gated skip connections that scale encoder feature maps based on decoder gating signals.
* **Joint Augmentation Pipeline:** A robust `JointTransform` class that simultaneously applies geometric transformations (flips, rotations, resizing) to both the input image and its corresponding targets.
* **Multi-Class Exploration:** Includes bonus setups for 3-class semantic mapping (foreground, background, boundary) and pre-trained classification backbones (MobileNetV2, EfficientNet, DenseNet).
* **Checkpoint & Resume:** Built-in validation monitoring that automatically saves the best-performing weights to Google Drive and resumes training seamlessly.
