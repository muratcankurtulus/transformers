import argparse
import math

import tiktoken
import torch
import torch._dynamo
import torch._inductor.config
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm

from gpt import GPT
from tokenizer import Tokenizer

torch._inductor.config.compile_threads = 1
torch._dynamo.config.suppress_errors = True


class ModelConfig(BaseModel):
    embed_dim: int = 384
    tgt_vocab_size: int = 384
    seq_len: int = 256
    num_layers: int = 6
    expansion_factor: int = 4
    n_heads: int = 6
    dropout_rate: float = 0.2


class DatasetConfig(BaseModel):
    batch_size: int = 64
    shuffle: bool = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len, tokenizer, tokenizer_type):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        if data_path.endswith((".pt", ".bin")):
            print(f"Loading pre-tokenized data from {data_path}")
            self.data = torch.load(data_path, weights_only=True)
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
            raise ValueError(f"Unsupported data file format: {data_path}. Please use .txt, .pt, or .bin")

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


def get_scheduler(optimizer, scheduler_type, warmup_steps, total_steps, min_lr=1e-6):
    """
    Creates a learning rate scheduler.

    Args:
        optimizer: The optimizer to use
        scheduler_type: Type of scheduler to use
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate

    Returns:
        A learning rate scheduler
    """
    if scheduler_type == "cosine":
        # Cosine schedule with warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay after warmup
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "linear":
        # Linear schedule with warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Linear decay after warmup
            return max(min_lr, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
            final_div_factor=1.0 / min_lr if min_lr > 0 else 25,
        )

    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=min_lr)

    else:
        # Default: no scheduler, constant learning rate
        return None


def main(
    tokenizer_path,
    train_data_path,
    eval_data_path,
    epochs,
    experiment_name,
    tokenizer_type,
    eval_interval,
    use_pretokenized,
    weight_decay,
    early_stopping_patience,
    dropout_rate,
):
    # Load tokenizer
    if tokenizer_type == "tiktoken":
        tokenizer = tiktoken.get_encoding("cl100k_base")  # or another model like "p50k_base"
    elif tokenizer_type == "default":
        # Only set tokenizer to None if BOTH files are pretokenized AND use_pretokenized flag is True
        needs_tokenizer = train_data_path.endswith(".txt") or eval_data_path.endswith(".txt")
        if needs_tokenizer:
            print(f"Loading tokenizer from {tokenizer_path} for text file processing")
            tokenizer = Tokenizer.load(tokenizer_path)
        else:
            tokenizer = None if use_pretokenized else Tokenizer.load(tokenizer_path)
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
        dropout_rate=dropout_rate,
    )
    model = GPT(**model_config.model_dump()).to("cuda")
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
    # Disable compilation as it's causing segmentation faults
    # try:
    #     model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    #     print("Model successfully compiled with torch.compile")
    # except Exception as e:
    #     print(f"Failed to compile model: {e}")
    #     print("Falling back to eager mode")
    print("Using eager mode (torch.compile disabled)")

    criterion = nn.CrossEntropyLoss()
    # Add weight decay to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-9)
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5, verbose=True)

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

    # Early stopping setup
    best_loss = float("inf")
    early_stopping_counter = 0
    best_model_path = f"{experiment_name}_best.pth"

    print(
        f"Training with dropout_rate={dropout_rate}, \
          weight_decay={weight_decay}, \
          early_stopping_patience={early_stopping_patience}"
    )

    # Training loop
    global_step = 0
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

                # Log loss and learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                tepoch.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")
                train_loss += loss.item()
                global_step += 1

                # Evaluate more frequently as training progresses
                # Start with normal interval, then reduce near the end
                dynamic_interval = max(100, eval_interval - (epoch * 50))

                if step % dynamic_interval == 0 and step != 0:
                    eval_loss = evaluate(model, criterion, eval_loader, model_config.tgt_vocab_size)
                    print(f"Step {global_step} | Eval Loss: {eval_loss:.4f} | LR: {current_lr:.6f}")

                    # Update scheduler
                    scheduler.step(eval_loss)

                    # Early stopping check
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        early_stopping_counter = 0
                        # Save best model
                        torch.save(model.state_dict(), best_model_path)
                        print(f"New best model saved! Loss: {best_loss:.4f}")
                    else:
                        early_stopping_counter += 1
                        print(f"No improvement for {early_stopping_counter} evaluations")
                        if early_stopping_counter >= early_stopping_patience:
                            print(
                                f"Early stopping triggered after {early_stopping_patience} evaluations without improvement"
                            )
                            print(f"Best eval loss: {best_loss:.4f}")
                            # Load best model for final evaluation
                            model.load_state_dict(torch.load(best_model_path))
                            return

            train_loss /= len(train_loader)

        # Epoch-level evaluation
        eval_loss = evaluate(model, criterion, eval_loader, model_config.tgt_vocab_size)
        print(
            f"Epoch {epoch} | \
              Train Loss: {train_loss:.4f} | \
              Eval Loss: {eval_loss:.4f} | \
              LR: {optimizer.param_groups[0]['lr']:.6f}\n"
        )

        # Update scheduler with epoch-level loss
        scheduler.step(eval_loss)

        # Early stopping check
        if eval_loss < best_loss:
            best_loss = eval_loss
            early_stopping_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Loss: {best_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} evaluations")
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} evaluations without improvement")
                print(f"Best eval loss: {best_loss:.4f}")
                break

        # Also save regular checkpoints
        if epoch % 5 == 0 and epoch != 0:
            torch.save(model.state_dict(), f"{experiment_name}_e{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}")

    print(f"Training completed. Best eval loss: {best_loss:.4f}")
    # Load best model for final use
    model.load_state_dict(torch.load(best_model_path))


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
        "--train_data",
        default="./toy_data/tiny_sp_train.txt",
        type=str,
        help="Path to the training data (.txt, .pt, or .bin)",
    )
    parser.add_argument(
        "--eval_data",
        default="./toy_data/tiny_sp_test.txt",
        type=str,
        help="Path to the evaluation data (.txt, .pt, or .bin)",
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
    parser.add_argument(
        "--eval_interval", type=int, default=500, help="Interval for evaluation during training (default: 500 steps)"
    )
    parser.add_argument(
        "--use_pretokenized", action="store_true", help="Flag to indicate input files are already tokenized"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer (default: 0.01)"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of evaluations with no improvement before early stopping (default: 5)",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for the model (default: 0.3)")

    args = parser.parse_args()

    main(
        args.tokenizer,
        args.train_data,
        args.eval_data,
        args.epochs,
        args.experiment_name,
        args.tokenizer_type,
        args.eval_interval,
        args.use_pretokenized,
        args.weight_decay,
        args.early_stopping_patience,
        args.dropout_rate,
    )
