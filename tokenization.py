import argparse

from tokenizer import Tokenizer


def remove_suffix(file_path):
    if "." in file_path:
        return file_path.rsplit(".", 1)[0]
    return file_path


def train_tokenizer(vocab_size, train_file_path):
    with open(train_file_path, encoding="utf-8") as f:
        text = f.read()

    print("training tokenizer...")
    tokenizer = Tokenizer(vocab_size=vocab_size)
    tokenizer.train(text)
    save_path = remove_suffix(train_file_path)
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}")


def encode_text(text):
    loaded_tokenizer = Tokenizer.load("toy_data/wiki_text_2")
    encoded = loaded_tokenizer.encode(text)
    print(f"Encoded: {encoded}")


def decode_text(encoded_text):
    loaded_tokenizer = Tokenizer.load("toy_data/wiki_text_2")
    decoded = loaded_tokenizer.decode(encoded_text)
    print(f"Decoded: {decoded}")


parser = argparse.ArgumentParser(description="Tokenizer utility")
parser.add_argument("--train", type=str, help="Path to training text file", required=False)
parser.add_argument("--encode", type=str, help="Text to encode")
parser.add_argument("--decode", type=str, help="Encoded text to decode")
parser.add_argument("--vocab_size", type=int, default=4096, help="Vocabulary size for training the tokenizer")

args = parser.parse_args()

if args.train:
    train_tokenizer(args.vocab_size, args.train)
elif args.encode:
    encode_text(args.encode)
elif args.decode:
    decode_text(args.decode)
else:
    print("Please provide an action: --train, --encode, or --decode")
