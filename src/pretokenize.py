import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from tokenizer import Tokenizer


def pretokenize_file(tokenizer_path, input_file, output_file=None, tokenizer_type="default"):
    """Pre-tokenize a text file and save it as a PyTorch tensor.

    Args:
        tokenizer_path: Path to the tokenizer or name for tiktoken
        input_file: Path to the input text file
        output_file: Path to save the output .pt file (defaults to input_file with .pt extension)
        tokenizer_type: Type of tokenizer to use ('default' or 'tiktoken')
    """
    # Set default output path if not provided
    if output_file is None:
        # Remove extension and add .pt
        output_file = str(Path(input_file).with_suffix(".pt"))

    print(f"Pre-tokenizing {input_file} to {output_file}...")

    # Load tokenizer
    if tokenizer_type == "tiktoken":
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
    else:
        tokenizer = Tokenizer.load(tokenizer_path)

    # Read input file
    with open(input_file, encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing data...")

    # Tokenize the data
    if tokenizer_type == "tiktoken":
        encoded_data = tokenizer.encode(text, allowed_special="all")
    else:
        encoded_data = tokenizer.encode(text)

    # Convert to tensor
    data_tensor = torch.tensor(encoded_data, dtype=torch.long)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save tensor to file
    torch.save(data_tensor, output_file)

    print(f"Tokenized data saved to {output_file}")
    print(f"Tensor shape: {data_tensor.shape}")
    print(f"Number of tokens: {len(data_tensor)}")


def batch_pretokenize(tokenizer_path, input_dir, output_dir=None, tokenizer_type="default", file_pattern="*.txt"):
    """Pre-tokenize all text files in a directory.

    Args:
        tokenizer_path: Path to the tokenizer or name for tiktoken
        input_dir: Directory containing text files to tokenize
        output_dir: Directory to save output .pt files (defaults to input_dir)
        tokenizer_type: Type of tokenizer to use ('default' or 'tiktoken')
        file_pattern: Pattern to match files (default: *.txt)
    """
    input_dir = Path(input_dir)

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Find all text files
    files = list(input_dir.glob(file_pattern))

    if not files:
        print(f"No files matching {file_pattern} found in {input_dir}")
        return

    print(f"Found {len(files)} files to tokenize")

    # Process each file
    for file_path in tqdm(files, desc="Processing files"):
        output_path = output_dir / file_path.with_suffix(".pt").name
        pretokenize_file(tokenizer_path, str(file_path), str(output_path), tokenizer_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-tokenize text files to PyTorch tensors for faster training")

    parser.add_argument(
        "--tokenizer",
        default="./toy_data/tiny_sp",
        type=str,
        help="Path to the tokenizer (for sentencepiece) or name (for tiktoken)",
    )

    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="default",
        choices=["default", "tiktoken"],
        help="Type of tokenizer to use (default: sentencepiece)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single file pre-tokenization
    file_parser = subparsers.add_parser("file", help="Pre-tokenize a single file")
    file_parser.add_argument("input_file", type=str, help="Path to the input text file")
    file_parser.add_argument(
        "--output_file", type=str, help="Path to save the output .pt file (defaults to input_file with .pt extension)"
    )

    # Batch pre-tokenization
    batch_parser = subparsers.add_parser("batch", help="Pre-tokenize all text files in a directory")
    batch_parser.add_argument("input_dir", type=str, help="Directory containing text files to tokenize")
    batch_parser.add_argument(
        "--output_dir", type=str, help="Directory to save output .pt files (defaults to input_dir)"
    )
    batch_parser.add_argument(
        "--file_pattern", type=str, default="*.txt", help="Pattern to match files (default: *.txt)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    elif args.command == "file":
        pretokenize_file(args.tokenizer, args.input_file, args.output_file, args.tokenizer_type)
    elif args.command == "batch":
        batch_pretokenize(args.tokenizer, args.input_dir, args.output_dir, args.tokenizer_type, args.file_pattern)
