import argparse

import torch

from gpt import GPT
from tokenizer import Tokenizer
from train_gpt import ModelConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT generate")
    parser.add_argument("--prompt", type=str, help="Prompt to generate text from", required=True)
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint", required=True)
    parser.add_argument("--length", type=int, default=100, help="Length of generated text")
    parser.add_argument("--tokenizer_path", type=str, help="Path to vocab file", required=True)
    parser.add_argument("--vocab_size", type=int, help="Vocab size", required=True)
    args = parser.parse_args()

    model_config = ModelConfig()
    model = GPT(**model_config.model_dump()).to("cuda")
    model.eval()

    tokenizer = Tokenizer.load(args.tokenizer_path)
    model.load_state_dict(torch.load(args.model_path))
    prompt = tokenizer.encode(args.prompt)
    generated = model.generate(prompt, args.length)
    generated_text = tokenizer.decode(generated)
    print(generated_text)
