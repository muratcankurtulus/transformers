# Description: Tokenization of text data
import collections
import regex as re
from typing import *
import joblib


def get_stats(tokens: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_tokens(tokens: List[Tuple[int, int]], pair: Tuple[int, int],
                 idx) -> List[Tuple[int, int]]:
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


def encode(text, merges: Dict[Tuple[int, int], int]) -> List[int]:
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge_tokens(tokens, pair, idx)
    return tokens


def decode(tokens: List[int], vocab: Dict[int, bytes]) -> str:
    tokens = b"".join([vocab[t] for t in tokens])
    text = tokens.decode("utf-8", errors="replace")
    return text


if __name__ == "__main__":
    with open("toy_data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    vocab = {idx: bytes([idx]) for idx in range(256)}

    vocab_size = 1024
    num_merges = vocab_size - 256

    pat = re.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
    lits_of_text = re.findall(pat, text)
    bytestrings = [item.encode("utf-8") for item in lits_of_text]
    text = b"".join(bytestrings)
    tokens = list(map(int, text))

    copy_tokens = tokens.copy()
    merges = {}
    for i in range(num_merges):
        stats = get_stats(copy_tokens)
        pair = max(stats, key=stats.get)
        print(f"Merge {pair} into idx -> {256 + i}")
        copy_tokens = merge_tokens(copy_tokens, pair, 256 + i)
        merges[pair] = 256 + i

    for (a, b), idx in merges.items():
        vocab[idx] = vocab[a] + vocab[b]

    print(f"\n\nMerges: {merges}")
    print(f"\n\nVocab: {vocab}")
    print(f"\n\nTokens: {tokens}")
    print(f"\n\nCopy Tokens: {copy_tokens}")
    print("tokens length:", len(tokens))
    print("ids length:", len(copy_tokens))
    print(f"compression ratio: {len(tokens) / len(copy_tokens):.2f}X")

    joblib.dump(merges, "toy_data/merges")
    joblib.dump(vocab, "toy_data/vocab")
    joblib.dump(copy_tokens, "toy_data/ids")
