import argparse

import tiktoken
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
    num_layers: int = 6
    expansion_factor: int = 4
    n_heads: int = 6


class DatasetConfig(BaseModel):
    batch_size: int = 64
    shuffle: bool = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len, tokenizer, tokenizer_type):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        if data_path.endswith(".pt"):
            print(f"Loading pre-tokenized data from {data_path}")
            self.data = torch.load(data_path)
        elif data_path.endswith(".txt"):
            print(f"Tokenizing data from {data_path}...")
            with open(data_path, encoding="utf-8") as f:
                data = f.read()

            if tokenizer_type == "tiktoken":
                encoded_data = self.tokenizer.encode(data, allowed_special="all")
            else:
                encoded_data = self.tokenizer.encode(data)

            self.data = torch.tensor(encoded_data)
            print("Tokenization complete.")
        else:
            raise ValueError(f"Unsupported data file format: {data_path}. Please use .txt or .pt")

        # Pre-calculate valid indices
        self.valid_indices = [i for i in range(len(self.data)) if i + self.seq_len + 1 <= len(self.data)]
        print(f"Found {len(self.valid_indices)} valid sequences.")

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
        for src, tgt in tepoch:
            src, tgt = src.cuda(non_blocking=True), tgt.cuda(non_blocking=True)
            mask = model.make_tgt_mask(tgt).cuda(non_blocking=True)
            output = model(src, mask)
            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
            total_loss += loss.item()
            tepoch.set_postfix(eval_loss=f"{loss.item():.4f}")

    model.train()
    return total_loss / len(eval_loader)


def main(tokenizer_path, train_data_path, eval_data_path, epochs, experiment_name, tokenizer_type):
    # Load tokenizer
    if tokenizer_type == "tiktoken":
        tokenizer = tiktoken.get_encoding("cl100k_base")  # or another model like "p50k_base"
    elif tokenizer_type == "default":
        # Tokenizer is only needed if we are tokenizing text files
        tokenizer = (
            Tokenizer.load(tokenizer_path)
            if train_data_path.endswith(".txt") or eval_data_path.endswith(".txt")
            else None
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

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
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
    #    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    # Load data and create DataLoader
    dataset_config = DatasetConfig(batch_size=args.batch_size, shuffle=args.shuffle)
    # Pass data paths directly to Dataset constructor
    train_dataset = Dataset(train_data_path, model_config.seq_len, tokenizer, tokenizer_type)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.shuffle,
        num_workers=4,
        pin_memory=True,
    )

    # Use shuffle=False for eval loader config
    eval_dataset_config = DatasetConfig(batch_size=args.batch_size, shuffle=False)  # Create separate config for eval
    eval_dataset = Dataset(eval_data_path, model_config.seq_len, tokenizer, tokenizer_type)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_dataset_config.batch_size,  # Use eval config batch size
        shuffle=eval_dataset_config.shuffle,  # Use eval config shuffle (False)
        num_workers=4,
        pin_memory=True,
    )

    # Training loop
    for epoch in range(epochs):
        train_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for step, (src, tgt) in enumerate(tepoch):
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
                if step % 500 == 0 and step != 0:
                    eval_step = evaluate(model, criterion, eval_loader, model_config.tgt_vocab_size)
                    print(f"Step {step} | Eval Loss: {eval_step}")
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
    parser.add_argument(
        "--tokenizer",
        default="./toy_data/tiny_sp",
        type=str,
        help="Path to the tokenizer (for sentencepiece) or name (for tiktoken). Only needed if data is in .txt format.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="default",
        choices=["default", "tiktoken"],
        help="Type of tokenizer to use (default: sentencepiece). Only needed if data is in .txt format.",
    )
    parser.add_argument(
        "--train_data", default="./toy_data/tiny_sp_train.txt", type=str, help="Path to the training data (.txt or .pt)"
    )
    parser.add_argument(
        "--eval_data", default="./toy_data/tiny_sp_test.txt", type=str, help="Path to the evaluation data (.txt or .pt)"
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

    main(args.tokenizer, args.train_data, args.eval_data, args.epochs, args.experiment_name, args.tokenizer_type)
