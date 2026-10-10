"""Single-GPU training with packed blocks, FP16, and a resumable wall-clock budget."""

import argparse
import hashlib
import json
import math
import struct
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gpt import GPT
from tokenizer import Tokenizer


class PackedTokens:
    """Memory-map BPE1 tokens; adjacent contexts advance by seq_len, rather than one token."""

    def __init__(self, path: Path, seq_len: int, tokenizer: Tokenizer):
        self.path = path
        self.seq_len = seq_len
        tokenizer.validate_saved_metadata(str(path) + ".tokenizer.json")
        with path.open("rb") as file:
            header = file.read(13)
        if len(header) != 13 or header[:4] != b"BPE1" or header[4] not in (2, 4):
            raise ValueError(f"Invalid BPE1 header: {path}")
        count = struct.unpack("<Q", header[5:])[0]
        if path.stat().st_size != 13 + count * header[4]:
            raise ValueError(f"Token count does not match file size: {path}")
        self.data = np.memmap(path, mode="r", dtype=f"<u{header[4]}", offset=13, shape=(count,))
        self.num_blocks = (count - 1) // seq_len
        if self.num_blocks < 1:
            raise ValueError(f"Need at least {seq_len + 1} tokens: {path}")
        # Record the data identity so a resume cannot silently switch corpora.
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        self.identity = {"sha256": digest.hexdigest(), "tokens": count}

    def batch(self, indices) -> tuple[torch.Tensor, torch.Tensor]:
        chunks = [
            self.data[int(index) * self.seq_len : int(index) * self.seq_len + self.seq_len + 1] for index in indices
        ]
        tokens = torch.from_numpy(np.array(chunks, dtype=np.int64))
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def autocast(device, precision):
    if device.type == "cuda" and precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def move_batch(batch, device):
    if device.type == "cuda":
        return tuple(tensor.pin_memory().to(device, non_blocking=True) for tensor in batch)
    return tuple(tensor.to(device) for tensor in batch)


@torch.no_grad()
def evaluate(model, data, batch_size, max_batches, device, precision):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for start in range(0, min(data.num_blocks, batch_size * max_batches), batch_size):
        indices = range(start, min(start + batch_size, data.num_blocks))
        inputs, targets = move_batch(data.batch(indices), device)
        with autocast(device, precision):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
        loss_sum += loss.item()
        token_count += targets.numel()
    model.train()
    return loss_sum / token_count


def scheduled_lr(step, warmup_steps, elapsed, hours, peak_lr):
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, max(0.0, elapsed / (hours * 3600)))
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def atomic_save(checkpoint, path):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this command on the GPU machine.")
    if device.type != "cuda" and args.precision != "fp32":
        raise ValueError("Use --precision fp32 for a CPU smoke test")
    if args.resume is None and any((args.output_dir / name).exists() for name in ("last.pth", "best.pth")):
        raise FileExistsError("Checkpoints already exist; use --resume or choose a new --output_dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    train_data = PackedTokens(args.train_data, args.seq_len, tokenizer)
    eval_data = PackedTokens(args.eval_data, args.seq_len, tokenizer)
    if train_data.identity == eval_data.identity:
        raise ValueError("Training and evaluation data must be different")
    model_config = {
        "embed_dim": args.embed_dim,
        "tgt_vocab_size": tokenizer.vocab_size,
        "num_layers": args.num_layers,
        "expansion_factor": 4,
        "n_heads": args.n_heads,
        "dropout_rate": args.dropout_rate,
    }
    training_config = {
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "precision": args.precision,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
    }
    model = GPT(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    generator = torch.Generator().manual_seed(args.seed)
    step, tokens_seen, elapsed_before, best_loss = 0, 0, 0.0, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        tokenizer.validate_metadata(saved.get("tokenizer"), str(args.resume))
        if saved["model_config"] != model_config or saved["training_config"] != training_config:
            raise ValueError("Resume requires the same model, batch, precision, optimizer, seed, and context settings")
        if saved["data"] != {"train": train_data.identity, "eval": eval_data.identity}:
            raise ValueError("Resume corpus differs from checkpoint corpus")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        generator.set_state(saved["sampling_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, tokens_seen = saved["step"], saved["tokens_seen"]
        elapsed_before, best_loss = saved["elapsed_seconds"], saved["best_loss"]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Parameters={parameter_count:,}; vocab={tokenizer.vocab_size:,}; "
        f"train_tokens={train_data.identity['tokens']:,}; packed_blocks={train_data.num_blocks:,}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(device)}; precision={args.precision}", flush=True)
    print("Random packed blocks sampled with replacement; tokens_seen counts training exposures, not unique tokens.")
    (args.output_dir / "config.json").write_text(
        json.dumps({"model": model_config, "training": training_config, "max_hours": args.max_hours}, indent=2) + "\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    last_save = started
    window_started, window_tokens = started, tokens_seen
    log_path = args.output_dir / "metrics.jsonl"

    def elapsed():
        return elapsed_before + time.monotonic() - started

    def checkpoint():
        return {
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer.metadata,
            "model_config": model_config,
            "training_config": training_config,
            "seq_len": args.seq_len,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "sampling_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "step": step,
            "tokens_seen": tokens_seen,
            "elapsed_seconds": elapsed(),
            "best_loss": best_loss,
            "data": {"train": train_data.identity, "eval": eval_data.identity},
        }

    def record_validation():
        nonlocal best_loss
        loss = evaluate(model, eval_data, args.batch_size, args.eval_batches, device, args.precision)
        print(f"step={step} eval_loss={loss:.4f} perplexity={math.exp(min(loss, 50)):.2f}", flush=True)
        metrics = {"step": step, "tokens_seen": tokens_seen, "eval_loss": loss, "hours": elapsed() / 3600}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(metrics) + "\n")
        if loss < best_loss:
            best_loss = loss
            atomic_save(checkpoint(), args.output_dir / "best.pth")

    model.train()
    try:
        while elapsed() < args.max_hours * 3600 and (args.max_steps is None or step < args.max_steps):
            optimizer.zero_grad(set_to_none=True)
            lr = scheduled_lr(step, args.warmup_steps, elapsed(), args.max_hours, args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_sum = 0.0
            for _ in range(args.gradient_accumulation):
                indices = torch.randint(train_data.num_blocks, (args.batch_size,), generator=generator)
                inputs, targets = move_batch(train_data.batch(indices), device)
                with autocast(device, args.precision):
                    logits = model(inputs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss; last completed checkpoint is preserved")
                loss_sum += loss.item()
                scaler.scale(loss / args.gradient_accumulation).backward()
                tokens_seen += targets.numel()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=args.precision == "fp32")
            scaler.step(optimizer)
            scaler.update()
            step += 1
            now = time.monotonic()
            if step % args.log_interval == 0:
                rate = (tokens_seen - window_tokens) / max(now - window_started, 1e-9)
                metrics = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": loss_sum / args.gradient_accumulation,
                    "tokens_per_second": rate,
                    "lr": lr,
                    "hours": elapsed() / 3600,
                }
                if device.type == "cuda":
                    metrics["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                print(json.dumps(metrics), flush=True)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps(metrics) + "\n")
                window_started, window_tokens = now, tokens_seen
            if step % args.eval_interval == 0:
                record_validation()
            if now - last_save >= args.checkpoint_seconds or step % args.eval_interval == 0:
                atomic_save(checkpoint(), args.output_dir / "last.pth")
                last_save = time.monotonic()
    except KeyboardInterrupt:
        print("Interrupted; saving current weights and optimizer state.", flush=True)
    record_validation()
    atomic_save(checkpoint(), args.output_dir / "last.pth")
    print(
        f"Finished: steps={step:,}, tokens_seen={tokens_seen:,}, hours={elapsed() / 3600:.2f}, best_loss={best_loss:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--eval_data", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max_hours", type=float, default=36, help="Total active run budget, including saved resumed time"
    )
    parser.add_argument("--max_steps", type=int, help="Optional total optimizer-step cap, useful for benchmarking")
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--checkpoint_seconds", type=float, default=600)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    positive = (
        args.max_hours,
        args.batch_size,
        args.gradient_accumulation,
        args.seq_len,
        args.embed_dim,
        args.num_layers,
        args.n_heads,
        args.eval_interval,
        args.eval_batches,
        args.checkpoint_seconds,
        args.log_interval,
        args.learning_rate,
    )
    if min(positive) <= 0 or args.warmup_steps < 0 or args.weight_decay < 0 or not 0 <= args.dropout_rate < 1:
        parser.error("Invalid training settings; sizes/intervals must be positive and dropout must be in [0, 1)")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be positive")
    if args.embed_dim % args.n_heads or (args.embed_dim // args.n_heads) % 2:
        parser.error("Head dimension must be an even integer for RoPE")
    train(args)


if __name__ == "__main__":
    main()
