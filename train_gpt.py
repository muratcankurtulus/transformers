import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel
from tqdm import tqdm

from gpt import GPT
from tokenizer import Tokenizer


class ModelConfig(BaseModel):
    embed_dim: int = 384
    tgt_vocab_size: int = 384
    seq_len: int = 256
    num_layers: int = 3
    expansion_factor: int = 2
    n_heads: int = 3


class DatasetConfig(BaseModel):
    batch_size: int = 64
    shuffle: bool = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data = torch.tensor(self.tokenizer.encode(data))
        # Pre-calculate valid indices
        self.valid_indices = [i for i in range(len(self.data)) if i + self.seq_len + 1 <= len(self.data)]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        src = self.data[real_idx : real_idx + self.seq_len]
        tgt = self.data[real_idx + 1 : real_idx + self.seq_len + 1]
        return src, tgt


@torch.no_grad()
def evaluate(model, criterion, eval_loader, vocab_size):
    model.eval()
    total_loss = 0
    with tqdm(eval_loader, unit="batch") as tepoch:
        with torch.no_grad():
            for src, tgt in tepoch:
                mask = model.make_tgt_mask(tgt).to("cuda")
                output = model(src, mask)
                loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                total_loss += loss.item()
                tepoch.set_postfix(eval_loss=f"{loss.item():.4f}")
    return total_loss / len(eval_loader)


def main(tokenizer_path, train_data_path, eval_data_path, epochs, experiment_name):
    # Load tokenizer
    tokenizer = Tokenizer.load(tokenizer_path)

    # Model configuration
    model_config = ModelConfig(
        embed_dim=args.embed_dim,
        tgt_vocab_size=args.tgt_vocab_size,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        expansion_factor=args.expansion_factor,
        n_heads=args.n_heads,
    )
    model = GPT(**model_config.model_dump()).to("cuda")
    #    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.1)

    # Load data and create DataLoader
    with open(train_data_path, encoding="utf-8") as f:
        data = f.read()

    with open(eval_data_path, encoding="utf-8") as f:
        data_eval = f.read()

    dataset_config = DatasetConfig(batch_size=args.batch_size, shuffle=args.shuffle)
    train_loader = torch.utils.data.DataLoader(
        Dataset(data, model_config.seq_len, tokenizer),
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.shuffle,
        num_workers=4,
        pin_memory=True,
    )

    dataset_config.shuffle = False
    eval_loader = torch.utils.data.DataLoader(
        Dataset(data_eval, model_config.seq_len, tokenizer),
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.shuffle,
        num_workers=4,
        pin_memory=True,
    )

    # Training loop
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for src, tgt in tepoch:
                src, tgt = src.cuda(non_blocking=True), tgt.cuda(non_blocking=True)
                mask = model.make_tgt_mask(tgt).cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                output = model(src, mask)
                loss = criterion(output.view(-1, model_config.tgt_vocab_size), tgt.view(-1))
                loss.backward()
                optimizer.step()
                # Log learning rate
                tepoch.set_postfix(loss=f"{loss.item():.4f}")
                train_loss += loss.item()

            train_loss /= len(train_loader)

        eval_loss = evaluate(model, criterion, eval_loader, model_config.tgt_vocab_size)
        print(f"Epoch {epoch} | Train Loss: {train_loss} | Eval Loss: {eval_loss}\n")
        if epoch % 5 == 0 and epoch != 0:
            torch.save(model.state_dict(), f"{experiment_name}_e{epoch}.pth")
            print("Model saved!")


if __name__ == "__main__":
    model_config = ModelConfig()
    dataset_config = DatasetConfig()

    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("--tokenizer", default="./toy_data/tiny_sp", type=str, help="Path to the tokenizer")
    parser.add_argument(
        "--train_data", default="./toy_data/tiny_sp_train.txt", type=str, help="Path to the training data"
    )
    parser.add_argument(
        "--eval_data", default="./toy_data/tiny_sp_test.txt", type=str, help="Path to the evaluation data"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=model_config.embed_dim,
        help=f"Embedding dimension (default: {model_config.embed_dim})",
    )
    parser.add_argument(
        "--tgt_vocab_size",
        type=int,
        default=model_config.tgt_vocab_size,
        help=f"Target vocabulary size (default: {model_config.tgt_vocab_size})",
    )
    parser.add_argument(
        "--seq_len", type=int, default=model_config.seq_len, help=f"Sequence length (default: {model_config.seq_len})"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=model_config.num_layers,
        help=f"Number of layers (default: {model_config.num_layers})",
    )
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=model_config.expansion_factor,
        help=f"Expansion factor (default: {model_config.expansion_factor})",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=model_config.n_heads,
        help=f"Number of attention heads (default: {model_config.n_heads})",
    )
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=dataset_config.batch_size,
        help=f"Batch size for training (default: {dataset_config.batch_size})",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=dataset_config.shuffle,
        help=f"Shuffle the dataset (default: {dataset_config.shuffle})",
    )

    args = parser.parse_args()

    main(args.tokenizer, args.train_data, args.eval_data, args.epochs, args.experiment_name)
