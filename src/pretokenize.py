import argparse
import os
import struct
import time
from pathlib import Path

import numpy as np
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


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def pretokenize_file_streaming(
    tokenizer_path: str,
    input_file: str,
    output_file: str = None,
    tokenizer_type: str = "default",
    output_format: str = "bin",
    chunk_size: int = 1000,
):
    """Pre-tokenize a text file in streaming fashion and save as .bin or .pt.

    This is the main recommended function - it streams input line by line,
    encodes in chunks, and writes output incrementally. Memory usage is bounded.

    Args:
        tokenizer_path: Path to the tokenizer or name for tiktoken
        input_file: Path to the input text file
        output_file: Path to save the output file (defaults based on output_format)
        tokenizer_type: Type of tokenizer to use ('default' or 'tiktoken')
        output_format: Output format - 'bin' (recommended) or 'pt'
        chunk_size: Number of lines to process at once
    """
    # Set default output path if not provided
    if output_file is None:
        suffix = ".bin" if output_format == "bin" else ".pt"
        output_file = str(Path(input_file).with_suffix(suffix))

    print(f"Pre-tokenizing {input_file} -> {output_file}")
    print(f"  Output format: {output_format}")

    # Load tokenizer
    if tokenizer_type == "tiktoken":
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        vocab_size = tokenizer.n_vocab
    else:
        try:
            resolved_path = resolve_tokenizer_path(tokenizer_path)
            print(f"  Loading tokenizer from {resolved_path}")
            tokenizer = Tokenizer.load(resolved_path)
            vocab_size = tokenizer.vocab_size
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure the tokenizer has been trained and exists at the specified path.")
            print("You can train a tokenizer using the tokenization.py script.")
            return

    # Determine dtype based on vocab size
    if vocab_size < 65536:
        dtype = np.uint16
        dtype_str = "uint16"
    else:
        dtype = np.uint32
        dtype_str = "uint32"
    print(f"  Vocab size: {vocab_size}, using dtype: {dtype_str}")

    # Get file size for progress tracking
    file_size = get_file_size(input_file)
    print(f"  Input file size: {file_size / 1024 / 1024:.2f} MB")

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    bytes_read = 0
    total_tokens = 0
    lines_processed = 0

    if output_format == "bin":
        # Stream directly to binary file
        with open(input_file, encoding="utf-8") as fin:
            with open(output_file, "wb") as fout:
                # Write header: magic bytes + dtype indicator + placeholder for token count
                # Header format: "BPE1" (4 bytes) + dtype (1 byte: 2=uint16, 4=uint32) + token_count (8 bytes)
                4 + 1 + 8
                fout.write(b"BPE1")  # Magic bytes
                fout.write(struct.pack("B", 2 if dtype == np.uint16 else 4))  # dtype size
                fout.write(struct.pack("<Q", 0))  # Placeholder for token count

                chunk_lines = []
                pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Tokenizing")

                for line in fin:
                    line_bytes = len(line.encode("utf-8"))
                    bytes_read += line_bytes
                    chunk_lines.append(line)
                    lines_processed += 1

                    if len(chunk_lines) >= chunk_size:
                        # Process chunk
                        chunk_text = "".join(chunk_lines)
                        if tokenizer_type == "tiktoken":
                            tokens = tokenizer.encode(chunk_text, allowed_special="all")
                        else:
                            tokens = tokenizer.encode(chunk_text)

                        # Write tokens
                        token_array = np.array(tokens, dtype=dtype)
                        fout.write(token_array.tobytes())
                        total_tokens += len(tokens)

                        pbar.update(sum(len(l.encode("utf-8")) for l in chunk_lines))
                        chunk_lines = []

                # Process remaining lines
                if chunk_lines:
                    chunk_text = "".join(chunk_lines)
                    if tokenizer_type == "tiktoken":
                        tokens = tokenizer.encode(chunk_text, allowed_special="all")
                    else:
                        tokens = tokenizer.encode(chunk_text)

                    token_array = np.array(tokens, dtype=dtype)
                    fout.write(token_array.tobytes())
                    total_tokens += len(tokens)
                    pbar.update(sum(len(l.encode("utf-8")) for l in chunk_lines))

                pbar.close()

                # Update header with actual token count
                fout.seek(5)  # Position after magic + dtype
                fout.write(struct.pack("<Q", total_tokens))

    else:
        # Legacy .pt format - still streaming but accumulates in memory at the end
        all_tokens = []

        with open(input_file, encoding="utf-8") as fin:
            chunk_lines = []
            pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Tokenizing")

            for line in fin:
                line_bytes = len(line.encode("utf-8"))
                bytes_read += line_bytes
                chunk_lines.append(line)
                lines_processed += 1

                if len(chunk_lines) >= chunk_size:
                    chunk_text = "".join(chunk_lines)
                    if tokenizer_type == "tiktoken":
                        tokens = tokenizer.encode(chunk_text, allowed_special="all")
                    else:
                        tokens = tokenizer.encode(chunk_text)

                    all_tokens.extend(tokens)
                    pbar.update(sum(len(l.encode("utf-8")) for l in chunk_lines))
                    chunk_lines = []

            # Process remaining
            if chunk_lines:
                chunk_text = "".join(chunk_lines)
                if tokenizer_type == "tiktoken":
                    tokens = tokenizer.encode(chunk_text, allowed_special="all")
                else:
                    tokens = tokenizer.encode(chunk_text)

                all_tokens.extend(tokens)
                pbar.update(sum(len(l.encode("utf-8")) for l in chunk_lines))

            pbar.close()

        total_tokens = len(all_tokens)
        data_tensor = torch.tensor(all_tokens, dtype=torch.long)
        torch.save(data_tensor, output_file)

    elapsed_time = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0

    print(f"\nTokenization completed in {elapsed_time:.2f} seconds")
    print(f"  Lines processed: {lines_processed}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Tokens/second: {tokens_per_sec:,.0f}")
    print(f"  Output saved to: {output_file}")

    # Report output file size
    output_size = os.path.getsize(output_file)
    print(f"  Output file size: {output_size / 1024 / 1024:.2f} MB")


def pretokenize_file(tokenizer_path, input_file, output_file=None, tokenizer_type="default"):
    """Legacy function - calls streaming version with .pt output for backwards compatibility."""
    pretokenize_file_streaming(
        tokenizer_path,
        input_file,
        output_file,
        tokenizer_type,
        output_format="pt",
    )


def batch_pretokenize(
    tokenizer_path,
    input_dir,
    output_dir=None,
    tokenizer_type="default",
    file_pattern="*.txt",
    output_format="bin",
):
    """Pre-tokenize all text files in a directory.

    Args:
        tokenizer_path: Path to the tokenizer or name for tiktoken
        input_dir: Directory containing text files to tokenize
        output_dir: Directory to save output files (defaults to input_dir)
        tokenizer_type: Type of tokenizer to use ('default' or 'tiktoken')
        file_pattern: Pattern to match files (default: *.txt)
        output_format: Output format - 'bin' or 'pt'
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
    else:
        resolved_path = tokenizer_path

    # Process each file
    for file_path in files:
        suffix = ".bin" if output_format == "bin" else ".pt"
        output_path = output_dir / file_path.with_suffix(suffix).name
        pretokenize_file_streaming(
            resolved_path,
            str(file_path),
            str(output_path),
            tokenizer_type,
            output_format,
        )


def read_bin_header(file_path: str) -> dict:
    """Read header from a .bin tokenized file.

    Args:
        file_path: Path to the .bin file

    Returns:
        Dict with 'dtype', 'token_count', 'data_offset'
    """
    with open(file_path, "rb") as f:
        magic = f.read(4)
        if magic != b"BPE1":
            raise ValueError(f"Invalid magic bytes in {file_path}: {magic}")

        dtype_size = struct.unpack("B", f.read(1))[0]
        token_count = struct.unpack("<Q", f.read(8))[0]

        dtype = np.uint16 if dtype_size == 2 else np.uint32

        return {
            "dtype": dtype,
            "token_count": token_count,
            "data_offset": 13,  # 4 + 1 + 8 bytes header
        }


def load_bin_file(file_path: str) -> np.ndarray:
    """Load a .bin tokenized file into a numpy array.

    For small files, this loads into memory.
    For training, prefer using numpy.memmap (see train_gpt.py).

    Args:
        file_path: Path to the .bin file

    Returns:
        Numpy array of token ids
    """
    header = read_bin_header(file_path)
    with open(file_path, "rb") as f:
        f.seek(header["data_offset"])
        data = np.frombuffer(f.read(), dtype=header["dtype"])
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-tokenize text files for faster training")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single file pre-tokenization
    file_parser = subparsers.add_parser("file", help="Pre-tokenize a single file")
    file_parser.add_argument("input_file", type=str, help="Path to the input text file")
    file_parser.add_argument("--output_file", type=str, help="Path to save the output file (defaults based on format)")
    file_parser.add_argument(
        "--tokenizer",
        default="./toy_data/tiny_sp",
        type=str,
        help="Path to the tokenizer (for default) or name (for tiktoken)",
    )
    file_parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="default",
        choices=["default", "tiktoken"],
        help="Type of tokenizer to use",
    )
    file_parser.add_argument(
        "--format",
        type=str,
        default="bin",
        choices=["bin", "pt"],
        help="Output format: 'bin' (memory-mappable, recommended) or 'pt' (PyTorch tensor)",
    )
    file_parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of lines to process at once (default: 1000)",
    )

    # Batch pre-tokenization
    batch_parser = subparsers.add_parser("batch", help="Pre-tokenize all text files in a directory")
    batch_parser.add_argument("input_dir", type=str, help="Directory containing text files to tokenize")
    batch_parser.add_argument("--output_dir", type=str, help="Directory to save output files (defaults to input_dir)")
    batch_parser.add_argument(
        "--file_pattern", type=str, default="*.txt", help="Pattern to match files (default: *.txt)"
    )
    batch_parser.add_argument(
        "--tokenizer",
        default="./toy_data/tiny_sp",
        type=str,
        help="Path to the tokenizer (for default) or name (for tiktoken)",
    )
    batch_parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="default",
        choices=["default", "tiktoken"],
        help="Type of tokenizer to use",
    )
    batch_parser.add_argument(
        "--format",
        type=str,
        default="bin",
        choices=["bin", "pt"],
        help="Output format: 'bin' (memory-mappable, recommended) or 'pt' (PyTorch tensor)",
    )

    # Info command to inspect .bin files
    info_parser = subparsers.add_parser("info", help="Show info about a .bin tokenized file")
    info_parser.add_argument("file", type=str, help="Path to the .bin file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    elif args.command == "file":
        pretokenize_file_streaming(
            args.tokenizer,
            args.input_file,
            args.output_file,
            args.tokenizer_type,
            args.format,
            args.chunk_size,
        )
    elif args.command == "batch":
        batch_pretokenize(
            args.tokenizer,
            args.input_dir,
            args.output_dir,
            args.tokenizer_type,
            args.file_pattern,
            args.format,
        )
    elif args.command == "info":
        try:
            header = read_bin_header(args.file)
            file_size = os.path.getsize(args.file)
            print(f"File: {args.file}")
            print(f"  Format: BPE1")
            print(f"  Token dtype: {header['dtype']}")
            print(f"  Token count: {header['token_count']:,}")
            print(f"  Data offset: {header['data_offset']} bytes")
            print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"Error reading file: {e}")
