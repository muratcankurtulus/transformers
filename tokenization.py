import os
from typing import Dict, List, Tuple

import joblib
import regex as re


class Tokenizer:
    """A simple tokenizer class for encoding and decoding text using byte pair
    encoding (BPE).

    Attributes:
        SPECIAL_TOKENS (Dict[str, int]): Special tokens with their corresponding IDs.
        vocab_size (int): The size of the vocabulary.
        vocab (Dict[int, bytes]): The vocabulary mapping token IDs to byte sequences.
        merges (Dict[Tuple[int, int], int]): The merge operations for BPE.
    """

    SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def __init__(self, vocab_size: int = 1024):
        """Initializes the Tokenizer with a given vocabulary size.

        Args:
            vocab_size (int): The size of the vocabulary. Default is 1024.
        """
        self.vocab_size = vocab_size
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.vocab.update({
            v: k.encode("utf-8")
            for k, v in self.SPECIAL_TOKENS.items()
        })
        self.merges = {}

    def get_stats(self,
                  tokens: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
        """Computes the frequency of each pair of consecutive tokens.

        Args:
            tokens (List[Tuple[int, int]]): The list of token pairs.

        Returns:
            Dict[Tuple[int, int], int]: A dictionary with token pairs as keys and their frequencies as values.
        """
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_tokens(self, tokens: List[Tuple[int, int]],
                     pair: Tuple[int, int], idx) -> List[Tuple[int, int]]:
        """Merges a specific pair of tokens in the token list.

        Args:
            tokens (List[Tuple[int, int]]): The list of tokens.
            pair (Tuple[int, int]): The pair of tokens to merge.
            idx (int): The index to assign to the merged token.

        Returns:
            List[Tuple[int, int]]: The new list of tokens after merging.
        """
        new_tokens = []
        a, b = pair
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def encode(self, text: str) -> List[int]:
        """Encodes a given text into a list of token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: The list of token IDs.
        """
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge_tokens(tokens, pair, idx)
        return tokens

    def add_special_tokens(self, tokens: List[int]) -> List[int]:
        """Adds special tokens to the list of token IDs.

        Args:
            tokens (List[int]): The list of token IDs.

        Returns:
            List[int]: The list of token IDs with special tokens added.
        """
        return [self.SPECIAL_TOKENS["<BOS>"]
                ] + tokens + [self.SPECIAL_TOKENS["<EOS>"]]

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of token IDs back into a string.

        Args:
            tokens (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS.values()]
        tokens = b"".join([self.vocab[t] for t in tokens])
        text = tokens.decode("utf-8", errors="replace")
        return text

    def train(self, text: str):
        """Trains the tokenizer on a given text to build the vocabulary and
        merge operations.

        Args:
            text (str): The input text to train on.
        """
        pat = re.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )
        lits_of_text = re.findall(pat, text)
        bytestrings = [item.encode("utf-8") for item in lits_of_text]
        text = b"".join(bytestrings)
        tokens = list(map(int, text))

        copy_tokens = tokens.copy()
        num_merges = self.vocab_size - 256 - len(self.SPECIAL_TOKENS)
        for i in range(num_merges):
            stats = self.get_stats(copy_tokens)
            pair = max(stats, key=stats.get)
            copy_tokens = self.merge_tokens(copy_tokens, pair,
                                            256 + i + len(self.SPECIAL_TOKENS))
            self.merges[pair] = 256 + i + len(self.SPECIAL_TOKENS)

        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    def save(self, directory: str):
        """Saves the tokenizer's vocabulary and merges to the specified
        directory.

        Args:
            directory (str): The directory where the tokenizer data will be saved.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(self.merges, os.path.join(directory, "merges"))
        joblib.dump(self.vocab, os.path.join(directory, "vocab"))

    @classmethod
    def load(cls, directory: str) -> 'Tokenizer':
        """Loads the tokenizer's vocabulary and merges from the specified
        directory.

        Args:
            directory (str): The directory from where the tokenizer data will be loaded.

        Returns:
            Tokenizer: The loaded Tokenizer instance.
        """
        merges = joblib.load(os.path.join(directory, "merges"))
        vocab = joblib.load(os.path.join(directory, "vocab"))
        tokenizer = cls()
        tokenizer.merges = merges
        tokenizer.vocab = vocab
        return tokenizer


# Example usage
if __name__ == "__main__":
    with open("toy_data/python_book.txt", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(vocab_size=4096)
    tokenizer.train(text)
    tokenizer.save("toy_data/python_book")

    loaded_tokenizer = Tokenizer.load("toy_data/python_book")

    encoded = loaded_tokenizer.encode("Hello, world!")
    print(f"Encoded: {encoded}")

    decoded = loaded_tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
