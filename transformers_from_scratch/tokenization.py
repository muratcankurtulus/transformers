import collections
import regex as re
from typing import *
import joblib

class Tokenizer:
    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3
    }

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.vocab.update({v: k.encode("utf-8") for k, v in self.SPECIAL_TOKENS.items()})
        self.merges = {}

    def get_stats(self, tokens: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_tokens(self, tokens: List[Tuple[int, int]], pair: Tuple[int, int], idx) -> List[Tuple[int, int]]:
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
        tokens = list(text.encode("utf-8"))
        tokens = [self.SPECIAL_TOKENS["<BOS>"]] + tokens + [self.SPECIAL_TOKENS["<EOS>"]]
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge_tokens(tokens, pair, idx)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS.values()]
        tokens = b"".join([self.vocab[t] for t in tokens])
        text = tokens.decode("utf-8", errors="replace")
        return text

    def train(self, text: str):
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
            copy_tokens = self.merge_tokens(copy_tokens, pair, 256 + i + len(self.SPECIAL_TOKENS))
            self.merges[pair] = 256 + i + len(self.SPECIAL_TOKENS)

        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

        joblib.dump(self.merges, "toy_data/merges")
        joblib.dump(self.vocab, "toy_data/vocab")
        joblib.dump(copy_tokens, "toy_data/ids")

# Example usage
if __name__ == "__main__":
    with open("toy_data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(vocab_size=1024)
    tokenizer.train(text)

    encoded = tokenizer.encode("Hello, world!")
    print(f"Encoded: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

