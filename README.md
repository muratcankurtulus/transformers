# GPT and Transformer Models

This project implements GPT and Transformer models from scratch using PyTorch, including custom tokenization, model architecture, training, and generation.

## Features

- Configurable GPT model with customizable parameters
- Custom BPE tokenizer with support for both SentencePiece and TikToken
- Support for Rotary Position Embeddings (RoPE)
- Efficient dataset management and training pipeline
- Text generation capabilities

## Requirements

- Python 3.10+
- Dependencies:
  - torch
  - pydantic
  - joblib
  - regex
  - tqdm
  - tiktoken (optional)
  - flash-attn (optional)

## Usage

### Training a Tokenizer

```bash
python src/tokenization.py --train ./path/to/corpus.txt --vocab_size 20000 --output_dir ./path/to/output
```

### Training a Model

```bash
python src/train_gpt.py \
    --tokenizer ./path/to/tokenizer \
    --tokenizer_type default \
    --train_data ./path/to/train_data.txt \
    --eval_data ./path/to/eval_data.txt \
    --epochs 50 \
    --embed_dim 384 \
    --tgt_vocab_size 10000 \
    --seq_len 256 \
    --num_layers 6 \
    --n_heads 6
```

### Generating Text

```bash
python src/generate.py \
    --prompt "Your text prompt" \
    --model_path path/to/model.pth \
    --tokenizer_path path/to/tokenizer \
    --vocab_size 10000 \
    --length 200
```

## Model Configuration

| Parameter          | Description                  | Default |
| ------------------ | ---------------------------- | ------- |
| `embed_dim`        | Embedding dimension          | `384`   |
| `tgt_vocab_size`   | Target vocabulary size       | `384`   |
| `seq_len`          | Sequence length              | `256`   |
| `num_layers`       | Number of transformer layers | `6`     |
| `expansion_factor` | Feedforward expansion factor | `4`     |
| `n_heads`          | Number of attention heads    | `6`     |

## Dataset Configuration

| Parameter    | Description         | Default |
| ------------ | ------------------- | ------- |
| `batch_size` | Batch size          | `64`    |
| `shuffle`    | Shuffle the dataset | `True`  |

## License

Open-source under the MIT License.
