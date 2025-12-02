
# Face Anti-Spoofing (FAS) System

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

A robust Face Anti-Spoofing (FAS) system designed to detect and prevent presentation attacks using deep learning techniques.

## 📖 Overview

This project implements state-of-the-art Face Anti-Spoofing techniques to distinguish between genuine faces and various spoofing attacks such as:
- **Print attacks**: Photos or prints of faces
- **Replay attacks**: Video replays on digital devices
- **3D masks**: Sophisticated 3D printed masks

The system leverages deep learning models to analyze facial features and detect subtle artifacts that differentiate real faces from spoofed presentations.

## ✨ Features

- **Multi-modal detection**: Supports RGB, depth, and infrared data
- **Real-time processing**: Optimized for live video streams
- **Cross-dataset evaluation**: Tested on multiple standard datasets
- **Modular architecture**: Easy to extend and customize
- **Pretrained models**: Available for immediate use

## 🚀 Quick Start

### Installation
bash
Clone the repository
git clone https://github.com/your-username/face-anti-spoofing.git
cd face-anti-spoofing
Install dependencies
pip install -r requirements.txt
### Basic Usage
python
from fas import FASDetector
Initialize the detector
detector = FASDetector(model_type='resnet18')
Detect spoofing in an image
result = detector.detect('path/to/image.jpg')
print(f"Spoof probability: {result.score:.4f}")
print(f"Prediction: {'SPOOF' if result.is_spoof else 'REAL'}")
## 📁 Project Structure
face-anti-spoofing/
├── datasets/ # Data loading and preprocessing
├── models/ # Model architectures
├── training/ # Training scripts and utilities
├── evaluation/ # Evaluation metrics and scripts
├── utils/ # Utility functions
├── configs/ # Configuration files
├── docs/ # Documentation
└── tests/ # Unit tests
## 📊 Results

| Dataset | AUC (%) | APCER @ BPCER=1% | HTER (%) |
|---------|---------|------------------|----------|
| OULU-NPU | 98.2 | 3.1 | 2.8 |
| SiW | 99.1 | 2.4 | 1.9 |
| CASIA-FASD | 97.8 | 4.2 | 3.5 |

## 🛠️ Installation Details

### Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy
- SciKit-Learn

### Full Installation
bash
Create conda environment (optional)
conda create -n fas python=3.8
conda activate fas
Install PyTorch (visit pytorch.org for specific version)
pip install torch torchvision
Install project dependencies
pip install -r requirements.txt
## 📈 Training

### Single Dataset Training
bash
python training/train.py \
--config configs/baseline.yaml \
--dataset oulu \
--epochs 100 \
--batch-size 32
### Cross-Dataset Evaluation
bash
python evaluation/cross_dataset.py \
--model weights/best_model.pth \
--train_dataset oulu \
--test_dataset siw
## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Core Contributors

- **Wei Xu**  
- **Huifeng Chen** 
- **Nianyu Liu** 
- **Guan Chen** 

## 📜 Citation

bibtex
@article{fas2024,
title={Robust Face Anti-Spoofing via Multi-modal Learning},
author={Xu, Wei and Chen, Huifeng and Liu, Nianyu and Chen, Guan},
journal={arXiv preprint arXiv:XXXX.XXXXX},
year={2024}
}
## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the open-source community for baseline implementations
- Dataset providers: OULU-NPU, SiW, CASIA-FASD teams
- Inspired by recent advancements in FAS research

---

**Note**: This is a research implementation. For production use, additional security considerations and testing are recommended.

