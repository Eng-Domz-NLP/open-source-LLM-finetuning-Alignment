# Open-Source LLM Fine-Tuning & Alignment

An end-to-end personal project for training and comparing language models across three alignment stages: full fine-tuning, QLoRA supervised fine-tuning, and DPO preference optimization.

Recommended public repository slug: `open-source-llm-finetuning-alignment`

## Overview

This repository contains a Colab-first notebook that walks through a complete alignment workflow using public Hugging Face datasets and open-source models. The goal was to build a practical, reproducible pipeline that starts from a strong baseline and then improves behavior through supervised and preference-based training.

The project was completed successfully. The final notebook includes the full training flow, evaluation utilities, and export helpers for saving the results in a structured way.

## What This Project Does

- Builds a full fine-tuning baseline with `FacebookAI/xlm-roberta-base`.
- Trains a QLoRA supervised fine-tuning stage with `Qwen/Qwen2-1.5B-Instruct`.
- Applies DPO alignment on the full `jondurbin/truthy-dpo-v0.1` training split.
- Compares the resulting models on a shared prompt suite.
- Logs experiments and outputs for later review.

## Why I Built It

The aim was to create a realistic alignment pipeline that demonstrates more than a single training run. I wanted to show how a model can move from a general baseline to a more instruction-following and preference-aligned system, while keeping the workflow practical for Colab and Kaggle constraints.

## Repository Contents

- `Final/open-source-llm-finetuning-alignment.ipynb` - main notebook containing the full pipeline.
- `open-source-llm-finetuning-alignment-colab.ipynb` - alternate notebook version.
- `open-source-llm-finetuning-alignment-part1-part2.ipynb`, `open-source-llm-finetuning-alignment-kaggle.ipynb`, `open-source-llm-finetuning-alignment-kaggle-v2.ipynb` - earlier working notebooks and variants.

## Pipeline Summary

### Part I - Full Fine-Tuning

Baseline training uses `FacebookAI/xlm-roberta-base` to establish a reference model for the assignment.

### Part II - SFT with QLoRA

Instruction-style training is performed with `Qwen/Qwen2-1.5B-Instruct` using 4-bit quantization and LoRA adapters to keep the run efficient on limited hardware.

### Part III - DPO Alignment

The preference optimization stage uses the full `jondurbin/truthy-dpo-v0.1` training split and evaluates several beta values to study the trade-off between helpfulness and refusal/safety behavior.

## Results

The project was successful: all three stages were designed, implemented, and evaluated as part of a complete alignment workflow. The notebook also includes a comparison section and export bundle helpers so the final artifacts can be reviewed and shared.

## Large Artifacts

The repository intentionally excludes large generated files such as trained model weights, adapter checkpoints, zip exports, and run artifacts. Those files are useful locally, but they are too large for a clean repository push and are regenerated from the notebook when needed.

If you need the outputs, run the notebook and regenerate them locally instead of committing them to version control.

## Recommended Run Order

1. Open `Final/open-source-llm-finetuning-alignment.ipynb` in Colab.
2. Run the setup and dataset cells.
3. Train Part I, Part II, and Part III in order.
4. Run the evaluation and comparison sections.
5. Export the final results bundle if you need submission artifacts.

## Suggested Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Datasets
- PEFT
- TRL
- bitsandbytes
- Weights & Biases

## Notes

- The notebook is designed for Colab/Kaggle-style GPU environments.
- Training and export paths are written to work cleanly across those runtimes.
- The repo is best presented as a personal ML engineering project rather than an assignment submission.
