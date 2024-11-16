import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gpt import GPT
from tokenizer import Tokenizer


class ModelConfig(BaseModel):
    embed_dim: int = 384
    tgt_vocab_size: int = 384 
    seq_len: int = 256
    num_layers: int = 4
    expansion_factor: int = 4
    n_heads: int = 6

class DatasetConfig(BaseModel):
    batch_size: int = 64
    shuffle: bool = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data = self.tokenizer.encode(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx + self.seq_len + 1 > len(self.data):
            idx = 0

        src = self.data[idx : idx + self.seq_len]
        tgt = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(src).to("cuda"), torch.tensor(tgt).to("cuda")

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
    writer = SummaryWriter(f"runs/{experiment_name}")

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


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
    )

    dataset_config.shuffle = False
    eval_loader = torch.utils.data.DataLoader(
        Dataset(data_eval, model_config.seq_len, tokenizer),
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.shuffle,
    )

    # Training loop
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for src, tgt in tepoch:
                mask = model.make_tgt_mask(tgt).to("cuda")
                optimizer.zero_grad()
                output = model(src, mask)
                loss = criterion(output.view(-1, model_config.tgt_vocab_size), tgt.view(-1))
                loss.backward()
                optimizer.step()

                # Log learning rate
                for param_group in optimizer.param_groups:
                    writer.add_scalar("Learning Rate", param_group["lr"], epoch * len(train_loader) + tepoch.n)

                # Log gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f"Gradients/{name}", param.grad, epoch * len(train_loader) + tepoch.n)
                tepoch.set_postfix(loss=f"{loss.item():.4f}")
                train_loss += loss.item()

            train_loss /= len(train_loader)

        # Log model parameters
        for name, param in model.named_parameters():
            writer.add_histogram(f"Parameters/{name}", param, epoch)

        eval_loss = evaluate(model, criterion, eval_loader, model_config.tgt_vocab_size)
        print(f"Epoch {epoch} | Train Loss: {train_loss} | Eval Loss: {eval_loss}\n")
        if epoch % 5 == 0 and epoch != 0:
            torch.save(model.state_dict(), f"{experiment_name}_e{epoch}.pth")
            print("Model saved!")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/eval", eval_loss, epoch)
    writer.flush()
    return writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer")
    parser.add_argument("--train_data", type=str, help="Path to the training data")
    parser.add_argument("--eval_data", type=str, help="Path to the evaluation data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension (default: 512)")
    parser.add_argument("--tgt_vocab_size", type=int, default=4096, help="Target vocabulary size (default: 4096)")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length (default: 256)")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers (default: 4)")
    parser.add_argument("--expansion_factor", type=int, default=4, help="Expansion factor (default: 4)")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle the dataset (default: True)")

    args = parser.parse_args()

    writer = main(args.tokenizer, args.train_data, args.eval_data, args.epochs, args.experiment_name)
    writer.close()
