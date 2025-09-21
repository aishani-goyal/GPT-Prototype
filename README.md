# GPT Prototype - Character-Level Language Model

A minimal GPT implementation built from scratch using PyTorch. Trains on Shakespeare text to generate similar character-level text using transformer architecture.

## Features

- Character-level tokenization
- Multi-head self-attention (6 heads, 6 layers)
- Positional encoding and layer normalization
- ~10.7M parameters

## Requirements

```bash
pip install torch
```

## Setup

1. Download the dataset:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

2. Open and run the notebook:
```bash
jupyter notebook gpt.ipynb
```

Or run all cells in the notebook environment of your choice (Jupyter Lab, Google Colab, VS Code, etc.)

## Model Configuration

- **Context length**: 256 characters
- **Embedding dimension**: 384
- **Attention heads**: 6
- **Transformer layers**: 6
- **Training iterations**: 5000

## What it does

1. Loads and preprocesses Shakespeare text
2. Creates character-to-integer mappings
3. Trains a transformer model to predict next characters
4. Generates new Shakespeare-like text

## Training Output

```
step 0: train loss 4.2234, val loss 4.2298
step 500: train loss 2.1234, val loss 2.1567
...
step 5000: train loss 1.5678, val loss 1.6234
```

## Generated Text Example

After training, the model generates text like:
```
DUKE OF AUMERLE:
What is the matter?

KING RICHARD II:
My lord, I have consider'd in my mind...
```

## Key Components

- **Head**: Single attention head
- **MultiHeadAttention**: Combines multiple attention heads  
- **Block**: Transformer block with attention + feed-forward
- **GPTLanguageModel**: Complete model architecture

## Customization

Modify hyperparameters at the top of the script:
- `n_embd`: Model size
- `n_layer`: Number of layers
- `block_size`: Context length
- `max_iters`: Training duration

This is an educational implementation to understand transformer architecture and autoregressive text generation.
