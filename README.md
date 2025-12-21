# GPT and Transformer Models

This project implements GPT and Transformer models from scratch using PyTorch, including custom tokenization, model architecture, training, and generation.

## Features

- Configurable GPT model with customizable parameters
- Custom BPE tokenizer with streaming training and fast encoding
- Memory-mapped `.bin` format for efficient large-scale training
- Support for Rotary Position Embeddings (RoPE)
- Efficient dataset management with minimal RAM overhead
- Text generation capabilities

## Requirements

- Python 3.10+
- Dependencies:
  - torch
  - numpy
  - pydantic
  - joblib
  - regex
  - tqdm
  - tiktoken (optional)
  - flash-attn (optional)

## Quick Start (Recommended Pipeline)

The recommended workflow for training on your own data:

### 1. Prepare your corpus

Create a text file with your training data (one document per line is ideal but not required).

### 2. Train a tokenizer (on a sample)

Train on a bounded sample to avoid memory issues:

```bash
python src/tokenization.py \
    --train ./data/train.txt \
    --vocab_size 8000 \
    --output_dir ./artifacts/tokenizer \
    --max_bytes 50000000  # ~50MB sample
```

Options:

- `--max_bytes`: Limit training data by bytes (recommended for large files)
- `--max_lines`: Limit training data by line count

### 3. Pretokenize to .bin format (streaming)

Convert text to memory-mappable binary format:

```bash
python src/pretokenize.py file ./data/train.txt \
    --tokenizer ./artifacts/tokenizer \
    --format bin

python src/pretokenize.py file ./data/eval.txt \
    --tokenizer ./artifacts/tokenizer \
    --format bin
```

This creates `train.bin` and `eval.bin` files that can be memory-mapped during training.

To inspect a .bin file:

```bash
python src/pretokenize.py info ./data/train.bin
```

### 4. Train the model

```bash
python src/train_gpt.py \
    --tokenizer ./artifacts/tokenizer \
    --train_data ./data/train.bin \
    --eval_data ./data/eval.bin \
    --experiment_name my_gpt \
    --epochs 50 \
    --embed_dim 384 \
    --tgt_vocab_size 8000 \
    --seq_len 256 \
    --num_layers 6 \
    --n_heads 6
```

### 5. Generate text

```bash
python src/generate.py \
    --prompt "Once upon a time" \
    --model_path ./my_gpt_best.pth \
    --tokenizer_path ./artifacts/tokenizer \
    --embed_dim 384 \
    --tgt_vocab_size 8000 \
    --seq_len 256 \
    --num_layers 6 \
    --n_heads 6 \
    --expansion_factor 4 \
    --dropout_rate 0.2 \
    --length 200
```

## Common Crawl Pipeline

For training on web-scale data from Common Crawl:

```bash
# 1. Download WET files (pre-extracted text)
cd data_prep
head -n 10 wet.paths | sed 's#^#https://data.commoncrawl.org/#' > wet_urls.txt
wget -c -i wet_urls.txt -P ./wet

# 2. Extract and filter English text
python exp_cc.py  # Outputs my_corpus_en.txt

# 3. Shuffle and split
shuf my_corpus_en.txt > corpus.shuf.txt
python -c "
from pathlib import Path
lines = Path('corpus.shuf.txt').read_text().splitlines(True)
n = int(len(lines) * 0.99)
Path('train.txt').write_text(''.join(lines[:n]))
Path('eval.txt').write_text(''.join(lines[n:]))
"

# 4. Train tokenizer on sample, pretokenize, and train model
# (follow steps 2-5 from Quick Start above)
```

## File Formats

| Format  | Extension | Description                    | Use Case                           |
| ------- | --------- | ------------------------------ | ---------------------------------- |
| Text    | `.txt`    | Raw text, tokenized on-the-fly | Small datasets, testing            |
| PyTorch | `.pt`     | Torch tensor                   | Legacy, moderate datasets          |
| Binary  | `.bin`    | Memory-mapped uint16/uint32    | **Recommended** for large datasets |

The `.bin` format includes a header with magic bytes, dtype info, and token count, followed by raw token data. This allows memory-mapping for O(1) RAM overhead regardless of dataset size.

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

## Benchmarking

To measure tokenization throughput:

```bash
python src/benchmark.py \
    --input ./data/sample.txt \
    --tokenizer ./artifacts/tokenizer \
    --iterations 3
```

This reports tokens/second and peak RAM usage.

## Architecture Notes

### Tokenizer (BPE)

- Uses GPT-2 style pre-tokenization regex
- Word-frequency based training (Counter-weighted pairs)
- Ranked-merge encoding with LRU cache for fast inference
- Streaming training for bounded memory usage

### Dataset

- Memory-maps `.bin` files using numpy.memmap
- No pre-computed index lists (O(1) memory overhead)
- Direct indexing for sequence extraction

## License

Open-source under the MIT License.
