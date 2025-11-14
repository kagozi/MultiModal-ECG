# ECG-Greedy: Multi-Modal ECG Classification with CWT Scalograms & Phasograms 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
<!-- [![Paper](https://img.shields.io/badge/ISBI-2025-blue)](https://biomedicalimaging.org/2025/)   -->
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

> **SOTA ensemble AUC 0.9238** on **PTB-XL Super-Diagnostic Task** using **early fusion of scalograms + phasograms + raw signals**.

This repository implements the **multi-representation ECG classification framework** from combining:
- **Time-domain**: XResNet1d101 (raw ECG)
- **Time-frequency**: CWT → **Scalograms**
- **Phase-domain**: CWT → **Phasograms**
- **Fusion**: Early & Late
- **Backbones**: Custom 2DCNN, ResNet50, EfficientNet-B0, Swin, **Hybrid Swin**

---

## Highlights
- **Ensemble AUC: 0.9238** (beats XResNet1d101: 0.9224)
- **Hybrid Swin** is the best single backbone (0.9023, early fusion)

### TABLE III. Performance of Top Models and Ensemble on the PTB-XL Superdiagnostic Task

| Domain        | Model           | Loss Function | AUC     | F1     |
|---------------|-----------------|---------------|---------|--------|
| Raw Signals   | XResNet1d101    | BCE           | 0.9224  | 0.7265 |
| Early Fusion  | Hybrid Swin     | Focal (W1)    | 0.9023  | 0.6970 |
| Early Fusion  | ResNet50        | Focal (W1)    | 0.8915  | 0.6850 |
| —             | **Ensemble**    | —             | **0.9238** | **0.7402** |

- **Early > Late > Single modality**
- Full preprocessing, training, and evaluation pipeline
- Reproducible on **PTB-XL** with patient-wise splits 

> **Note:** All the commands are based on a Unix based system.
> For a different system look for similar commands for it.


## Setup

We are using Python version 3.11.9

```bash
$ python --version
Python 3.11.9
```
### Requirements

```bash
# Clone and install
git clone https://github.com/kagozi/MultiModal-ECG.git
cd MultiModal-ECG

pip install -r requirements.txt
```
### Python virtual environment

**Create** a virtual environment:

```bash
python3 -m venv .venv
```

`.venv` is the name of the folder that would contain the virtual environment.

**Activate** the virtual environment:

```bash
source .venv/bin/activate
```

**Windows**
```bash
source .venv/Scripts/activate
```
# Quick Start
## Download the PTBXL dataset to a preferred directory: 
```bash
wget https://physionet.org/content/ptb-xl/get-zip/1.0.3/
```

## Run full pipeline: preprocess → train → evaluate → ensemble

```bash
python3 load_and_standardize.py # creates standardized raw signal representations
python3 generate_cwt.py # creates Wavelet representations (Phasogram and Scalograms)
python3 benchmark.py # Trains XResNet1d101 on raw signals
python3 train_models.py # Trains CWT based models on Phasogram and Scalograms
python3 tests_ensemble.py # Run tests, generate confusion matrices, and evaluate different ensembling strategies
```




