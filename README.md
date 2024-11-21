# GPT Model Training

This project trains a GPT model using PyTorch to process text data. It includes tokenization, dataset preparation, and a training loop with evaluation.

## Features

- Implements a configurable GPT model using `torch.nn` for text sequence generation.
- Utilizes `torch.utils.data` for efficient dataset management.
- Includes logging with TensorBoard for monitoring training progress.
- Supports dynamic configuration for model and dataset parameters using Pydantic.

## Requirements

- Python 3.10+
- Dependencies:
  - `torch`
  - `tqdm`
  - `pydantic`
  - `torchvision`
  - `tensorboard`

Install dependencies using `pip`:
```bash 
pip install torch torchvision pydantic tqdm tensorboard
```

## File Structure

- `blocks.py`: Defines the building blocks of the GPT model.
- `gpt.py`: Defines the GPT model.
- `tokenizer.py`: Defines Tokenizer class for encoding and decoding text.
- `tokoenization.py`: Defines functions for tokenization and dataset preparation.
- `train_gpt.py`: Main script for training and evaluation.

## Usage

### Command-Line Arguments

| Argument             | Description                                                      | Default                        |
| -------------------- | ---------------------------------------------------------------- | ------------------------------ |
| `--tokenizer`        | Path to the tokenizer file                                       | `./toy_data/tiny_sp`           |
| `--train_data`       | Path to the training data                                        | `./toy_data/tiny_sp_train.txt` |
| `--eval_data`        | Path to the evaluation data                                      | `./toy_data/tiny_sp_test.txt`  |
| `--epochs`           | Number of training epochs                                        | `100`                          |
| `--embed_dim`        | Embedding dimension                                              | `384`                          |
| `--tgt_vocab_size`   | Target vocabulary size                                           | `384`                          |
| `--seq_len`          | Sequence length                                                  | `256`                          |
| `--num_layers`       | Number of transformer layers                                     | `3`                            |
| `--expansion_factor` | Feedforward expansion factor                                     | `2`                            |
| `--n_heads`          | Number of attention heads                                        | `3`                            |
| `--experiment_name`  | Name of the experiment (logs saved under `runs/experiment_name`) | None                           |
| `--batch_size`       | Training batch size                                              | `64`                           |
| `--shuffle`          | Shuffle dataset                                                  | `True`                         |

### Running the Training Script

```
python train.py --tokenizer ./path/to/tokenizer --train_data ./path/to/train.txt --eval_data ./path/to/eval.txt --epochs 50 --experiment_name my_experiment
```

### Output

- Logs are saved in the `runs/` directory for visualization in TensorBoard.
- Model checkpoints are saved every 5 epochs as `<experiment_name>_e<epoch>.pth`.

### TensorBoard Visualization

Start TensorBoard:
``` tensorboard --logdir=runs```
Open [http://localhost:6006](http://localhost:6006) in your browser to monitor training.

## Configurations

### Model Config (`ModelConfig`)

| Parameter          | Description                  | Default |
| ------------------ | ---------------------------- | ------- |
| `embed_dim`        | Embedding dimension          | `384`   |
| `tgt_vocab_size`   | Target vocabulary size       | `384`   |
| `seq_len`          | Sequence length              | `256`   |
| `num_layers`       | Number of transformer layers | `3`     |
| `expansion_factor` | Feedforward expansion factor | `2`     |
| `n_heads`          | Number of attention heads    | `3`     |

### Dataset Config (`DatasetConfig`)

| Parameter    | Description         | Default |
| ------------ | ------------------- | ------- |
| `batch_size` | Batch size          | `64`    |
| `shuffle`    | Shuffle the dataset | `True`  |

## Customization

You can modify `gpt.py` or `tokenizer.py` and/or `tokenization.py` to use a different model architecture or tokenizer setup as needed.

## License

This project is open-source and licensed under the MIT License.
