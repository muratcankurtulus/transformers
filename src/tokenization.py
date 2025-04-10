import argparse
import os

from tokenizer import Tokenizer


def remove_suffix(file_path):
    if "." in file_path:
        return file_path.rsplit(".", 1)[0]
    return file_path


def train_tokenizer(vocab_size, train_file_path, output_dir=None):
    with open(train_file_path, encoding="utf-8") as f:
        text = f.read()

    print("Training tokenizer...")
    tokenizer = Tokenizer(vocab_size=vocab_size)
    tokenizer.train(text)

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


parser = argparse.ArgumentParser(description="Tokenizer utility")
parser.add_argument("--train", type=str, help="Path to training text file", required=False)
parser.add_argument("--output_dir", type=str, help="Directory to save tokenizer files", required=False)
parser.add_argument("--encode", type=str, help="Text to encode")
parser.add_argument("--decode", type=str, help="Encoded text to decode")
parser.add_argument("--tokenizer", type=str, default="toy_data/wiki_text_2", help="Path to the tokenizer")
parser.add_argument("--vocab_size", type=int, default=4096, help="Vocabulary size for training the tokenizer")

args = parser.parse_args()

if args.train:
    train_tokenizer(args.vocab_size, args.train, args.output_dir)
elif args.encode:
    encode_text(args.encode, args.tokenizer)
elif args.decode:
    decode_text(args.decode, args.tokenizer)
else:
    print("Please provide an action: --train, --encode, or --decode")
