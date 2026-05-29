# circMAC: Multi-scale Circular Architecture for circRNA Analysis

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

**circMAC** is a specialized deep learning framework designed for circular RNA (circRNA) sequence modeling. It introduces a multi-scale circular architecture that effectively captures local, sequential, and global dependencies unique to circular genomic structures.

## 🚀 Key Features

- **Proposed Method: circMAC**: A hybrid architecture combining the strengths of Attention, Mamba (SSM), and Convolutional networks.
- **Circular-Aware Design**: 
  - **Circular Relative Bias**: Attention mechanism that understands the continuous loop of circRNAs.
  - **Circular Padding CNN**: Local feature extraction without boundary artifacts.
- **Multi-scale Processing**: Captures biological signals across different sequence lengths using global down/up sampling.
- **Comprehensive Self-Supervised Learning**: Supports pretraining with MLM, SSP (Secondary Structure), and BPP (Base Pairing Probability).
- **Targeted Fine-tuning**: Optimized for precise miRNA target site prediction.

## 🏗 Architecture

The core of circMAC is the `CircMACBlock`, which features a 3-branch routing mechanism:
1. **Attention Branch**: Global dependency modeling with circular relative bias.
2. **Mamba Branch**: Efficient sequential modeling via State Space Models.
3. **CNN Branch**: Local pattern extraction with circular padding.

## 📁 Project Structure

```text
.
├── models/
│   ├── circmac.py       # Proposed CircMAC architecture
│   ├── heads.py         # Specialized heads (UnifiedSiteHead, etc.)
│   └── model.py         # Model wrapper and backbone management
├── pretraining.py       # Self-supervised pretraining entry point
├── training.py          # Supervised fine-tuning entry point (Sites task)
├── trainer.py           # Central training logic & orchestrator
├── utils.py             # Data processing and metrics
└── GEMINI.md            # Detailed technical context for AI agents
```

## 🛠 Usage

### 1. Installation

```bash
git clone https://github.com/juseong03/circmac.git
cd circmac
pip install -r requirements.txt
```

### 2. Pretraining (Proposed Method)

Learn circRNA representations using MLM, SSP, and BPP:

```bash
python pretraining.py \
    --model_name circmac \
    --mlm \
    --ssp \
    --pairing \
    --epochs 300 \
    --device 0
```

### 3. Downstream Fine-tuning (miRNA Target Sites)

Fine-tune the pretrained model for specific interaction site prediction:

```bash
python training.py \
    --model_name circmac \
    --target mirna \
    --task sites \
    --load_pretrained <your_experiment_name> \
    --epochs 100 \
    --device 0
```

## 📊 Methodology

### Pretraining Tasks
- **MLM (Masked Language Modeling)**: Predicts masked nucleotides.
- **SSP (Secondary Structure Prediction)**: Predicts RNA folding in dot-bracket notation.
- **BPP (Base Pairing Probability)**: Predicts the 2D base-pairing interaction matrix.

### Fine-tuning Task
- **Sites Prediction**: Per-position prediction of miRNA binding sites on the circRNA sequence, utilizing the circular nature of the molecule.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
