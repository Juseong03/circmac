# circMAC: Multi-scale Circular Architecture for circRNA

circMAC is the **proposed method** in this project, specifically designed for circular RNA (circRNA) sequence analysis. It leverages a multi-scale circular architecture to capture both local and global dependencies in circular genomic sequences. The framework supports self-supervised pretraining followed by fine-tuning on downstream miRNA target site prediction.

## Project Terminology

To ensure consistency during development, use the following standardized terms:

- **circMAC:** The proposed hybrid multi-scale model architecture tailored for circular RNAs.
- **Pretraining Tasks:**
  - **MLM (Masked Language Modeling):** Self-supervised task masking and predicting nucleotide tokens.
  - **SSP (Secondary Structure Prediction):** Task predicting the 2차 구조 (dot-bracket notation, e.g., `(`, `)`, `.`) of the RNA.
  - **BPP / Pairing (Base Pairing Probability):** Task predicting the 2D interaction matrix of nucleotide base pairs.
- **Architectural Components:**
  - **Circular Relative Bias:** Attention bias that calculates distance based on a circle (e.g., distance between start and end is 1), rather than a straight line.
  - **Circular CNN (Circular Padding):** Convolutional layer that wraps padding around the sequence ends, treating the sequence as a continuous loop.
  - **Mamba:** The State Space Model (SSM) branch used for efficient sequential processing.
  - **Router:** The gating mechanism in `CircMACBlock_v3` that dynamically weights and fuses the Attention, Mamba, and CNN branches.
- **Downstream Task:**
  - **Sites:** The primary downstream task focusing on predicting specific interaction sites on the circRNA sequence.
- **Downstream Target:**
  - **miRNA:** microRNA.

## Core Method: circMAC

- **Architecture:** A hybrid multi-scale model featuring:
  - **Circular Relative Bias:** Custom attention mechanism tailored for circular sequences.
  - **Multi-scale Processing:** Captures features across different sequence lengths.
  - **Flexible Branches:** Combines Attention, Mamba, and Convolutional branches (configurable via ablation flags).
- **Implementation:** Found in `models/circmac.py` and wrapped via `models/model.py`.

## Key Pretraining Tasks

The framework supports self-supervised pretraining with a focus on the following tasks:

1.  **MLM (Masked Language Modeling):** Predicting masked nucleotides to learn sequence representation.
2.  **SSP (Secondary Structure Prediction):** Predicting the secondary structure (dot-bracket tokens) of the RNA.
3.  **BPP (Base Pairing Probability / Pairing):** Predicting the pairing matrix to understand the spatial structure.

## Key Files & Directories

- `training.py`: Main entry point for supervised fine-tuning (Sites task).
- `pretraining.py`: Main entry point for self-supervised pretraining (MLM, SSP, Pairing).
- `trainer.py`: Central `Trainer` class managing loops, logging, and evaluation.
- `models/circmac.py`: Implementation of the proposed **circMAC** model.
- `utils_config.py`: Configuration for circMAC and baseline models.
- `data/`: Directory for input datasets (`.pkl`, `.json`).

## Usage Guide

### 1. Pretraining (Proposed Workflow)
Run `pretraining.py` with the specific tasks: MLM, SSP, and Pairing.

```bash
python pretraining.py \
    --model_name circmac \
    --mlm \
    --ssp \
    --pairing \
    --epochs 300 \
    --device 0 \
    --verbose
```

### 2. Downstream Fine-tuning
Run `training.py` to fine-tune the (pretrained) circMAC model on the **sites** task.

```bash
python training.py \
    --model_name circmac \
    --target mirna \
    --task sites \
    --load_pretrained <experiment_name> \
    --epochs 100 \
    --device 0
```

## Development Conventions

- **Proposed Method Priority:** When adding features or running benchmarks, prioritize the `circmac` model.
- **Reproducibility:** Default seed is 42. Experiment results are logged under `logs/circmac/`.
- **Ablation Studies:** Use flags like `--no_attn`, `--no_mamba`, or `--no_conv` to analyze circMAC components.
