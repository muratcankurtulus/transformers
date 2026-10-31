import argparse

import torch

from blocks import POS_ENCODING_TYPES
from devices import DEVICE_HELP, resolve_device
from gpt import GPT
from tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with a matching tokenizer and checkpoint")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    for name in ("embed_dim", "n_heads", "num_layers", "expansion_factor", "seq_len", "tgt_vocab_size"):
        parser.add_argument(f"--{name}", type=int, help="Override saved config; required for older checkpoints")
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--pos_encoding_type", choices=POS_ENCODING_TYPES, help="Override saved config")
    parser.add_argument("--length", type=int, default=100, help="Number of new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Zero selects greedy decoding")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help=DEVICE_HELP)
    args = parser.parse_args()
    if args.length < 0:
        parser.error("--length must be nonnegative")
    tokenizer = Tokenizer.load(args.tokenizer_path)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    tokenizer.validate_metadata(checkpoint.get("tokenizer"), args.model_path)
    config = dict(checkpoint.get("model_config", {}))
    for name in ("embed_dim", "n_heads", "num_layers", "expansion_factor", "dropout_rate", "tgt_vocab_size"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
        if name not in config:
            parser.error(f"Checkpoint has no model configuration; supply --{name}")
    seq_len = args.seq_len if args.seq_len is not None else checkpoint.get("seq_len")
    if seq_len is None or seq_len < 1:
        parser.error("Supply a positive --seq_len for the checkpoint's training context")
    if args.pos_encoding_type is not None:
        config["pos_encoding_type"] = args.pos_encoding_type
    config.setdefault("max_seq_len", seq_len)
    tokenizer.validate_vocab_size(config["tgt_vocab_size"])
    torch.manual_seed(args.seed)
    model = GPT(**config).to(resolve_device(args.device)).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    # The prepared corpus has text separators, rather than explicit special-token IDs.
    generated = model.generate(
        tokenizer.encode(args.prompt),
        args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        context_length=seq_len,
        forbidden_token_ids=list(tokenizer.SPECIAL_TOKENS.values()),
    )
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
