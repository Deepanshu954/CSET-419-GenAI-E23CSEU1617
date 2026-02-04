# CSET419 - Week 4 Lab: Text Generation

This repository contains implementations for various text generation models as part of the Week 4 Lab.

## Overview
The goal is to understand and implement different approaches to text generation, ranging from statistical models to advanced neural networks.

## Architecture
- **`main.py`**: **Run this file** to access the interactive menu for all models.
- **`utils.py`**: Handles data loading and vocabulary consistent across models.
- **`ngram_gen.py`**: N-gram (Trigram) model implementation.
- **`rnn_gen.py`**: Simple RNN model implementation.
- **`lstm_gru_gen.py`**: LSTM and GRU model implementations.
- **`transformer_gen.py`**: Transformer Encoder model implementation.
- **`dataset.txt`**: The text corpus.

## Requirements
- Python 3.x
- PyTorch
- NumPy

Install dependencies:
```bash
pip install torch numpy
```

## How to Run

**Method 1: Interactive Menu (Recommended)**
This allows you to train a model once and then generate text multiple times without retraining.
```bash
python main.py
```
1. Select a model (e.g., "1. N-gram Model").
2. Select "1. Train Model" (only need to do this once per model).
3. Select "2. Generate Text" to enter a seed word and get output.

**Method 2: Standalone Scripts**
You can still run individual scripts to train (and save) the model.
```bash
python ngram_gen.py
python rnn_gen.py
python lstm_gru_gen.py
python transformer_gen.py
```

## Model Files
Trained models are saved to disk so they can be reused:
- `ngram_model.pkl`
- `rnn_model.pth`
- `lstm_model.pth`
- `gru_model.pth`
- `transformer_model.pth`
- `vocab.pkl` (Vocabulary mapping)
