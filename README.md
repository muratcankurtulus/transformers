# GPT and Transformer Models

This project implements GPT and Transformer models from scratch using PyTorch. It includes custom tokenization, model architectures, and training pipelines.

## Features

- Custom GPT and Transformer implementations with configurable architectures
- Byte-Pair Encoding (BPE) tokenizer with special token support
- Support for both Rotary Position Embeddings (RoPE) and sinusoidal positional encodings
- Efficient dataset management with PyTorch DataLoader
- Dynamic configuration using Pydantic
- Text generation capabilities

## Requirements

- Python 3.11+
- Dependencies:
  - `torch>=2.4.1`
  - `pydantic`
  - `joblib`
  - `regex`
  - `tqdm`
  - `flash-attn`

Install dependencies using `pip`:

## Usage

### Training

Train a GPT model using the training script:

### Model Features

#### Tokenizer

- Custom BPE tokenizer implementation with caching
- Support for special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`)
- Compatible with both custom tokenizer and tiktoken

#### Architecture

- Configurable GPT and Transformer models
- Support for both Rotary Position Embeddings (RoPE) and sinusoidal encodings
- Multi-head attention with dropout and layer normalization
- Configurable feed-forward expansion factor

### Training Features

- Model checkpoints saved every 5 epochs
- Evaluation during training (every 500 steps)
- Support for both CPU and CUDA training
- Progress tracking with tqdm
- Efficient data loading with PyTorch DataLoader

### Model Configurations

#### GPT Config (`ModelConfig`)

| Parameter          | Description                  | Default |
| ------------------ | ---------------------------- | ------- |
| `embed_dim`        | Embedding dimension          | `384`   |
| `tgt_vocab_size`   | Target vocabulary size       | `384`   |
| `seq_len`          | Sequence length              | `256`   |
| `num_layers`       | Number of transformer layers | `6`     |
| `expansion_factor` | Feedforward expansion factor | `4`     |
| `n_heads`          | Number of attention heads    | `6`     |

#### Dataset Config (`DatasetConfig`)

| Parameter    | Description         | Default |
| ------------ | ------------------- | ------- |
| `batch_size` | Batch size          | `64`    |
| `shuffle`    | Shuffle the dataset | `True`  |

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is open-source and licensed under the MIT License.
