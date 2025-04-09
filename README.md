# GPT and Transformer Models

\<\<\<\<\<\<\< HEAD
This project trains a GPT model using PyTorch for text generation. It includes support for different tokenizers, dataset preparation, a training loop with evaluation, and a script for text generation.

## Features

- Implements a configurable GPT model using `torch.nn`.
- Utilizes `torch.utils.data` for efficient dataset management.
- Supports dynamic configuration for model and dataset parameters using Pydantic.
- Supports both custom SentencePiece tokenizers (via `tokenizer.py`) and TikToken tokenizers.
- Includes a script (`generate.py`) to generate text using a trained model.

## Requirements

- Python 3.10+ (tested with 3.11)
- Dependencies listed in `pyproject.toml`:
  - `torch`
  - `pydantic`
  - `joblib` (likely used by `tokenizer.py`)
  - `regex` (likely used by `tokenizer.py`)
  - `tqdm`
  - `tiktoken` (optional, for TikToken tokenizer)
  - `flash-attn` (optional, likely for performance)

Install dependencies using `pip`:

```bash
pip install torch pydantic joblib regex tqdm tiktoken flash-attn
# Or install from pyproject.toml if using a package manager like Poetry or pip>=21.1
# pip install .
```

## File Structure

- `src/blocks.py`: Defines the building blocks of the Transformer model (e.g., Attention, FeedForward).
- `src/gpt.py`: Defines the GPT model architecture using the blocks.
- `src/tokenizer.py`: Defines a custom `Tokenizer` class (likely SentencePiece-based).
- `src/tokenization.py`: Contains utility functions related to tokenization (confirm specific usage if needed).
- `src/transformer.py`: Defines a basic Transformer model (potentially for comparison or alternative use).
- `src/train_gpt.py`: Main script for training and evaluating the GPT model.
- `src/generate.py`: Script for generating text using a trained model checkpoint.
- `pyproject.toml`: Project configuration and dependencies.
- `toy_data/`: Directory containing example data and tokenizer files.
- `runs/`: Default directory for TensorBoard logs (if enabled) or other experiment outputs.

## Usage

### Training (`train_gpt.py`)

#### Command-Line Arguments

| Argument             | Description                                                         | Default                        |
| -------------------- | ------------------------------------------------------------------- | ------------------------------ |
| `--tokenizer`        | Path to the custom tokenizer file or name of the TikToken tokenizer | `./toy_data/tiny_sp`           |
| `--tokenizer_type`   | Type of tokenizer (`default` for custom/SentencePiece, `tiktoken`)  | `default`                      |
| `--train_data`       | Path to the training data file                                      | `./toy_data/tiny_sp_train.txt` |
| `--eval_data`        | Path to the evaluation data file                                    | `./toy_data/tiny_sp_test.txt`  |
| `--epochs`           | Number of training epochs                                           | `100`                          |
| `--embed_dim`        | Embedding dimension                                                 | `384`                          |
| `--tgt_vocab_size`   | Target vocabulary size (should match tokenizer)                     | `384`                          |
| `--seq_len`          | Sequence length                                                     | `256`                          |
| `--num_layers`       | Number of transformer layers                                        | `6`                            |
| `--expansion_factor` | Feedforward expansion factor                                        | `4`                            |
| `--n_heads`          | Number of attention heads                                           | `6`                            |
| `--experiment_name`  | Name of the experiment (used for saving checkpoints)                | None                           |
| `--batch_size`       | Training batch size                                                 | `64`                           |
| `--shuffle`          | Shuffle dataset (boolean flag)                                      | `True`                         |

#### Running the Training Script

```bash
python src/train_gpt.py \
    --tokenizer ./toy_data/tiny_sp \
    --tokenizer_type default \
    --train_data ./toy_data/tiny_sp_train.txt \
    --eval_data ./toy_data/tiny_sp_test.txt \
    --epochs 50 \
    --experiment_name my_gpt_experiment \
    --embed_dim 384 \
    --tgt_vocab_size 10000 \ # Adjust based on your tokenizer
    --seq_len 256 \
    --num_layers 6 \
    --n_heads 6 \
    --batch_size 32
```

### Generation (`generate.py`)

#### Command-Line Arguments

| Argument           | Description                        | Required | Default |
| ------------------ | ---------------------------------- | -------- | ------- |
| `--prompt`         | Initial text prompt                | Yes      | -       |
| `--model_path`     | Path to the trained model `.pth`   | Yes      | -       |
| `--tokenizer_path` | Path to the custom tokenizer file  | Yes      | -       |
| `--vocab_size`     | Vocabulary size of the model       | Yes      | -       |
| `--length`         | Number of tokens to generate       | No       | `100`   |
| `--embed_dim`      | Embedding dim used during training | No       | `160`   |
| `--n_heads`        | Num heads used during training     | No       | `2`     |
| `--num_layers`     | Num layers used during training    | No       | `3`     |

#### Running the Generation Script

```bash
python src/generate.py \
    --prompt "Once upon a time" \
    --model_path my_gpt_experiment_e50.pth \
    --tokenizer_path ./toy_data/tiny_sp \
    --vocab_size 10000 \ # Should match training
    --length 200
```

*Note: The `generate.py` script currently has hardcoded model configuration parameters (`embed_dim`, `n_heads`, `num_layers`). Ensure these match the model specified in `--model_path` or modify the script.*

### Training the Custom Tokenizer (`src/tokenization.py`)

If you want to use the custom BPE tokenizer (`tokenizer_type='default'` in `train_gpt.py`), you first need to train it on your corpus using the provided `src/tokenization.py` script.

This script uses the `Tokenizer` class from `src/tokenizer.py` and handles training, saving, encoding, and decoding via command-line arguments.

**Running Tokenizer Training:**

Use the `--train` argument to specify the path to your text corpus and `--vocab_size` to set the desired vocabulary size.

```bash
python src/tokenization.py \
    --train ./path/to/your/corpus.txt \
    --vocab_size 20000
```

This will train the tokenizer and save the necessary `vocab` and `merges` files to a directory named after your input file (without the extension) in the same location (e.g., `./path/to/your/corpus`).

You can then point the main training script (`train_gpt.py`) to this directory using `--tokenizer ./path/to/your/corpus --tokenizer_type default`.

The `src/tokenization.py` script also supports encoding (`--encode "text"`) and decoding (`--decode "[tokens]"`) using a pre-trained tokenizer (note: the script currently hardcodes the path `toy_data/wiki_text_2` for encode/decode, you might need to adjust it).

### Output

- Model checkpoints are saved periodically during training (e.g., every 5 epochs by default in `train_gpt.py`) as `<experiment_name>_e<epoch>.pth` in the project root directory.
- Generated text from `generate.py` is printed to the console.
  \=======
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

> > > > > > > f58c1d9f371e7e012d26935191af213c1cc99c19

- Configurable GPT and Transformer models
- Support for both Rotary Position Embeddings (RoPE) and sinusoidal encodings
- Multi-head attention with dropout and layer normalization
- Configurable feed-forward expansion factor

\<\<\<\<\<\<\< HEAD
Default configurations are defined within `train_gpt.py` using Pydantic `BaseModel`s. They can be overridden via command-line arguments.

### Model Config (`ModelConfig` in `train_gpt.py`)

\=======

### Training Features

- Model checkpoints saved every 5 epochs
- Evaluation during training (every 500 steps)
- Support for both CPU and CUDA training
- Progress tracking with tqdm
- Efficient data loading with PyTorch DataLoader

### Model Configurations

#### GPT Config (`ModelConfig`)

> > > > > > > f58c1d9f371e7e012d26935191af213c1cc99c19

| Parameter          | Description                  | Default |
| ------------------ | ---------------------------- | ------- |
| `embed_dim`        | Embedding dimension          | `384`   |
| `tgt_vocab_size`   | Target vocabulary size       | `384`   |
| `seq_len`          | Sequence length              | `256`   |
| `num_layers`       | Number of transformer layers | `6`     |
| `expansion_factor` | Feedforward expansion factor | `4`     |
| `n_heads`          | Number of attention heads    | `6`     |

\<\<\<\<\<\<\< HEAD

### Dataset Config (`DatasetConfig` in `train_gpt.py`)

\=======

#### Dataset Config (`DatasetConfig`)

> > > > > > > f58c1d9f371e7e012d26935191af213c1cc99c19

| Parameter    | Description         | Default |
| ------------ | ------------------- | ------- |
| `batch_size` | Batch size          | `64`    |
| `shuffle`    | Shuffle the dataset | `True`  |

## Contributing

\<\<\<\<\<\<\< HEAD

- Modify model architecture in `src/gpt.py` or `src/blocks.py`.
- Implement or use different tokenizers by adjusting `train_gpt.py` and potentially `src/tokenizer.py`.
- Change training loop, optimizer, or loss function in `src/train_gpt.py`.
  \=======
  Feel free to submit issues and enhancement requests.

> > > > > > > f58c1d9f371e7e012d26935191af213c1cc99c19

## License

This project is open-source and licensed under the MIT License.
