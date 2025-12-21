import argparse
import os
from typing import Iterator

from tokenizer import Tokenizer


def remove_suffix(file_path):
    if "." in file_path:
        return file_path.rsplit(".", 1)[0]
    return file_path


def stream_text_file(file_path: str, max_bytes: int = None, max_lines: int = None) -> Iterator[str]:
    """Stream lines from a text file with optional limits.

    Args:
        file_path: Path to the text file.
        max_bytes: Maximum bytes to read (approximate, stops after line exceeds).
        max_lines: Maximum lines to read.

    Yields:
        Lines from the file.
    """
    bytes_read = 0
    lines_read = 0

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            yield line

            lines_read += 1
            bytes_read += len(line.encode("utf-8"))

            if max_lines is not None and lines_read >= max_lines:
                print(f"Reached max_lines limit ({max_lines})")
                break
            if max_bytes is not None and bytes_read >= max_bytes:
                print(f"Reached max_bytes limit ({max_bytes} bytes)")
                break

    print(f"Read {lines_read} lines, {bytes_read / 1024 / 1024:.2f} MB")


def train_tokenizer(
    vocab_size: int,
    train_file_path: str,
    output_dir: str = None,
    max_bytes: int = None,
    max_lines: int = None,
) -> Tokenizer:
    """Train a tokenizer on a text file with optional sampling limits.

    Args:
        vocab_size: Target vocabulary size.
        train_file_path: Path to training text file.
        output_dir: Directory to save tokenizer files.
        max_bytes: Maximum bytes to use for training (for bounded memory).
        max_lines: Maximum lines to use for training.

    Returns:
        Trained Tokenizer instance.
    """
    print("Training tokenizer...")
    print(f"  vocab_size: {vocab_size}")
    if max_bytes:
        print(f"  max_bytes: {max_bytes / 1024 / 1024:.1f} MB")
    if max_lines:
        print(f"  max_lines: {max_lines}")

    tokenizer = Tokenizer(vocab_size=vocab_size)

    # Use streaming training with bounded input
    text_iterator = stream_text_file(train_file_path, max_bytes=max_bytes, max_lines=max_lines)
    tokenizer.train_streaming(text_iterator)

    # Determine save path
    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        save_path = output_dir
    else:
        # Default behavior: save in same location as training file with suffix removed
        save_path = remove_suffix(train_file_path)
        os.makedirs(save_path, exist_ok=True)

    # Save tokenizer files (merges and vocab files required by pretokenize.py)
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}")
    print("Created tokenizer files:")
    for file in os.listdir(save_path):
        print(f"  - {os.path.join(save_path, file)}")

    print("\nNOTE: These files can now be used with pretokenize.py to tokenize your text data.")

    return tokenizer


def encode_text(text, tokenizer_path):
    loaded_tokenizer = Tokenizer.load(tokenizer_path)
    encoded = loaded_tokenizer.encode(text)
    print(f"Encoded: {encoded}")


def decode_text(encoded_text, tokenizer_path):
    loaded_tokenizer = Tokenizer.load(tokenizer_path)
    decoded = loaded_tokenizer.decode(encoded_text)
    print(f"Decoded: {decoded}")


def main():
    parser = argparse.ArgumentParser(description="Tokenizer utility")
    parser.add_argument("--train", type=str, help="Path to training text file", required=False)
    parser.add_argument("--output_dir", type=str, help="Directory to save tokenizer files", required=False)
    parser.add_argument("--encode", type=str, help="Text to encode")
    parser.add_argument("--decode", type=str, help="Encoded text to decode")
    parser.add_argument("--tokenizer", type=str, default="toy_data/wiki_text_2", help="Path to the tokenizer")
    parser.add_argument("--vocab_size", type=int, default=4096, help="Vocabulary size for training the tokenizer")

    # New sampling arguments for bounded memory training
    parser.add_argument(
        "--max_bytes",
        type=int,
        default=None,
        help="Maximum bytes to read from training file (e.g., 50000000 for ~50MB). "
        "Use this to train on a sample without loading entire file.",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Maximum lines to read from training file. " "Use this to train on a sample without loading entire file.",
    )

    args = parser.parse_args()

    if args.train:
        train_tokenizer(
            args.vocab_size,
            args.train,
            args.output_dir,
            max_bytes=args.max_bytes,
            max_lines=args.max_lines,
        )
    elif args.encode:
        encode_text(args.encode, args.tokenizer)
    elif args.decode:
        decode_text(args.decode, args.tokenizer)
    else:
        print("Please provide an action: --train, --encode, or --decode")


if __name__ == "__main__":
    main()
