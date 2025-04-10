import argparse
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from tokenizer import Tokenizer


def resolve_tokenizer_path(tokenizer_path):
    """Resolve the tokenizer path to handle both directory and file paths.

    Args:
        tokenizer_path: Path to the tokenizer

    Returns:
        The resolved tokenizer path
    """
    # Check if the path exists as a directory
    if os.path.isdir(tokenizer_path):
        return tokenizer_path

    # Check if the path exists with /merges and /vocab
    if os.path.exists(os.path.join(tokenizer_path, "merges")):
        return tokenizer_path

    # Check if it's a file path with extension
    path = Path(tokenizer_path)
    if path.suffix:
        # Try without extension
        base_path = str(path.with_suffix(""))
        if os.path.exists(os.path.join(base_path, "merges")):
            return base_path

    # The path might be a base path without the directory structure
    if os.path.exists(tokenizer_path + "/merges"):
        return tokenizer_path

    raise FileNotFoundError(
        f"Could not find tokenizer files at {tokenizer_path}. "
        f"Expected to find 'merges' and 'vocab' files in this directory."
    )


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
        try:
            # Try to resolve the tokenizer path
            resolved_path = resolve_tokenizer_path(tokenizer_path)
            print(f"Loading tokenizer from {resolved_path}")
            tokenizer = Tokenizer.load(resolved_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure the tokenizer has been trained and exists at the specified path.")
            print("You can train a tokenizer using the tokenization.py script.")
            return

    # Read input file
    with open(input_file, encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing data...")

    # Tokenize the data
    print(f"Text length: {len(text)} characters")
    print("This may take a while for large files. Starting tokenization...")

    # Initialize progress bar to track tokenization progress (0-100%)
    progress_bar = tqdm(total=100, desc="Tokenizing", unit="%")

    # Start tracking time
    start_time = time.time()

    # Add a progress update function to handle progress reporting
    def progress_callback(processed_chars):
        progress_bar.update(processed_chars)

    if tokenizer_type == "tiktoken":
        encoded_data = tokenizer.encode(text, allowed_special="all")
    else:
        # Call encode with progress monitoring
        encoded_data = tokenizer.encode(text, progress_callback=progress_callback)

    # Close the progress bar
    progress_bar.close()

    # Report tokenization time
    elapsed_time = time.time() - start_time
    print(f"Tokenization completed in {elapsed_time:.2f} seconds")

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

    # First try to resolve the tokenizer path once for all files
    if tokenizer_type != "tiktoken":
        try:
            resolved_path = resolve_tokenizer_path(tokenizer_path)
            print(f"Using tokenizer from {resolved_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure the tokenizer has been trained and exists at the specified path.")
            print("You can train a tokenizer using the tokenization.py script.")
            return

    # Process each file
    for file_path in tqdm(files, desc="Processing files"):
        output_path = output_dir / file_path.with_suffix(".pt").name
        if tokenizer_type == "tiktoken":
            pretokenize_file(tokenizer_path, str(file_path), str(output_path), tokenizer_type)
        else:
            pretokenize_file(resolved_path, str(file_path), str(output_path), tokenizer_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-tokenize text files to PyTorch tensors for faster training")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single file pre-tokenization
    file_parser = subparsers.add_parser("file", help="Pre-tokenize a single file")
    file_parser.add_argument("input_file", type=str, help="Path to the input text file")
    file_parser.add_argument(
        "--output_file", type=str, help="Path to save the output .pt file (defaults to input_file with .pt extension)"
    )
    file_parser.add_argument(
        "--tokenizer",
        default="./toy_data/tiny_sp",
        type=str,
        help="Path to the tokenizer (for sentencepiece) or name (for tiktoken)",
    )
    file_parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="default",
        choices=["default", "tiktoken"],
        help="Type of tokenizer to use (default: sentencepiece)",
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
    batch_parser.add_argument(
        "--tokenizer",
        default="./toy_data/tiny_sp",
        type=str,
        help="Path to the tokenizer (for sentencepiece) or name (for tiktoken)",
    )
    batch_parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="default",
        choices=["default", "tiktoken"],
        help="Type of tokenizer to use (default: sentencepiece)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    elif args.command == "file":
        pretokenize_file(args.tokenizer, args.input_file, args.output_file, args.tokenizer_type)
    elif args.command == "batch":
        batch_pretokenize(args.tokenizer, args.input_dir, args.output_dir, args.tokenizer_type, args.file_pattern)
