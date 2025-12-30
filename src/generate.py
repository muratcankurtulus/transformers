import argparse

import torch

from gpt import GPT
from tokenizer import Tokenizer
from train_gpt import ModelConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT generate")
    parser.add_argument("--prompt", type=str, help="Prompt to generate text from", required=True)
    parser.add_argument("--embed_dim", type=int, help="Embed dim", required=True)
    parser.add_argument("--n_heads", type=int, help="Number of heads", required=True)
    parser.add_argument("--num_layers", type=int, help="Number of layers", required=True)
    parser.add_argument("--expansion_factor", type=int, help="Expansion factor", required=True)
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate", required=True)
    parser.add_argument("--seq_len", type=int, help="Sequence length", required=True)
    parser.add_argument("--tgt_vocab_size", type=int, help="Target vocabulary size", required=True)
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint", required=True)
    parser.add_argument("--length", type=int, default=100, help="Length of generated text")
    parser.add_argument("--tokenizer_path", type=str, help="Path to vocab file", required=True)
    args = parser.parse_args()

    model_config = ModelConfig(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        expansion_factor=args.expansion_factor,
        dropout_rate=args.dropout_rate,
        seq_len=args.seq_len,
        tgt_vocab_size=args.tgt_vocab_size,
    )
    # Exclude seq_len from model config - GPT doesn't use it
    gpt_params = {k: v for k, v in model_config.model_dump().items() if k != "seq_len"}
    model = GPT(**gpt_params).to("cuda")
    model.eval()

    tokenizer = Tokenizer.load(args.tokenizer_path)
    model.load_state_dict(torch.load(args.model_path))
    prompt = tokenizer.encode(args.prompt)
    generated = model.generate(prompt, args.length)
    generated_text = tokenizer.decode(generated)
    print(generated_text)
