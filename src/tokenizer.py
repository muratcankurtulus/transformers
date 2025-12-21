import os
from collections import Counter
from functools import lru_cache
from typing import Dict, Iterator, List, Tuple, Union

import joblib
import regex as re
import torch
from tqdm import tqdm


class Tokenizer:
    """A simple tokenizer class for encoding and decoding text using Byte Pair Encoding (BPE).

    Attributes:
        SPECIAL_TOKENS (Dict[str, int]): Special tokens with their corresponding IDs.
        BASE_VOCAB_SIZE (int): The size of the base vocabulary (256 bytes).
        vocab_size (int): The size of the vocabulary.
        vocab (Dict[int, bytes]): The vocabulary mapping token IDs to byte sequences.
        merges (Dict[Tuple[int, int], int]): The merge operations for BPE.
    """

    SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    BASE_VOCAB_SIZE = 256

    # GPT-2 style pre-tokenization regex pattern
    # Splits text into words, numbers, punctuation, and whitespace
    _GPT2_PAT = re.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )

    def __init__(self, vocab_size: int = 1024, cache_size: int = 10000):
        """Initializes the Tokenizer with a given vocabulary size.

        Args:
            vocab_size (int): The size of the vocabulary. Default is 1024.
            cache_size (int): Size of LRU cache for encoding pieces. Default is 10000.
        """
        self.vocab_size = vocab_size
        self.vocab = {idx: bytes([idx]) for idx in range(self.BASE_VOCAB_SIZE)}
        self.vocab.update({v: k.encode("utf-8") for k, v in self.SPECIAL_TOKENS.items()})
        self.merges: Dict[Tuple[int, int], int] = {}
        self._merge_ranks: Dict[Tuple[int, int], int] = {}  # pair -> rank (lower = merge first)
        self._cache_size = cache_size
        self._encode_piece_cached = None  # Will be set after load/train

    def _build_merge_ranks(self) -> None:
        """Build merge_ranks from merges dict. Lower rank = higher priority merge."""
        # merges is {pair: token_id}; token_id order corresponds to merge order
        # We want pair -> rank where rank is the order in which merges were learned
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        self._merge_ranks = {pair: rank for rank, (pair, _) in enumerate(sorted_merges)}
        # Rebuild cached encode function with new ranks
        self._build_encode_cache()

    def _build_encode_cache(self) -> None:
        """Build/rebuild the LRU-cached piece encoder."""
        merge_ranks = self._merge_ranks

        @lru_cache(maxsize=self._cache_size)
        def encode_piece_cached(piece_bytes: bytes) -> Tuple[int, ...]:
            """Encode a single piece (pre-token) to token ids using ranked BPE merges."""
            if len(piece_bytes) == 0:
                return ()

            # Start with byte-level tokens
            tokens = list(piece_bytes)

            # Apply merges until no more can be applied
            while len(tokens) >= 2:
                # Find the pair with lowest rank (highest priority)
                best_pair = None
                best_rank = float("inf")
                best_idx = -1

                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in merge_ranks:
                        rank = merge_ranks[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            best_idx = i

                if best_pair is None:
                    # No more merges possible
                    break

                # Apply the merge
                merged_token = self.merges[best_pair]
                tokens = tokens[:best_idx] + [merged_token] + tokens[best_idx + 2 :]

            return tuple(tokens)

        self._encode_piece_cached = encode_piece_cached

    def encode(self, text: str, progress_callback=None) -> List[int]:
        """Encodes a given text into a list of token IDs using ranked BPE merges.

        This is a fast O(n * m) encoder where n is text length and m is number of merges,
        with LRU caching for repeated pre-tokens (common in natural language).

        Args:
            text (str): The input text to encode.
            progress_callback: Optional callback function for progress tracking.
                              The function takes one parameter: percentage points to add.

        Returns:
            List[int]: The list of token IDs.
        """
        if self._encode_piece_cached is None:
            self._build_merge_ranks()

        # Split text into pre-tokens using GPT-2 regex
        pieces = re.findall(self._GPT2_PAT, text)
        total_pieces = len(pieces)

        result = []
        last_progress = 0

        for i, piece in enumerate(pieces):
            piece_bytes = piece.encode("utf-8")
            tokens = self._encode_piece_cached(piece_bytes)
            result.extend(tokens)

            # Report progress
            if progress_callback and total_pieces > 0:
                current_progress = int(100 * (i + 1) / total_pieces)
                if current_progress > last_progress:
                    progress_callback(current_progress - last_progress)
                    last_progress = current_progress

        # Ensure we report 100% at the end
        if progress_callback and last_progress < 100:
            progress_callback(100 - last_progress)

        return result

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts efficiently (shares cache).

        Args:
            texts: List of texts to encode.

        Returns:
            List of token id lists.
        """
        return [self.encode(text) for text in texts]

    def get_stats(self, tokens: List[int]) -> Dict[Tuple[int, int], int]:
        """Computes the frequency of each pair of consecutive tokens.

        Args:
            tokens (List[int]): The list of tokens.

        Returns:
            Dict[Tuple[int, int], int]: A dictionary with token pairs as keys and their frequencies as values.
        """
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_tokens(self, tokens: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """Merges a specific pair of tokens in the token list.

        Args:
            tokens (List[int]): The list of tokens.
            pair (Tuple[int, int]): The pair of tokens to merge.
            idx (int): The index to assign to the merged token.

        Returns:
            List[int]: The new list of tokens after merging.
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

    def add_special_tokens(self, tokens: List[int]) -> List[int]:
        """Adds special tokens to the list of token IDs.

        Args:
            tokens (List[int]): The list of token IDs.

        Returns:
            List[int]: The list of token IDs with special tokens added.
        """
        return [self.SPECIAL_TOKENS["<BOS>"]] + tokens + [self.SPECIAL_TOKENS["<EOS>"]]

    def decode(self, tokens: Union[List[int], torch.Tensor]) -> str:
        """Decodes a list of token IDs back into a string.

        Args:
            tokens (Union[List[int], torch.Tensor]): The list of token IDs or tensor to decode.

        Returns:
            str: The decoded string.
        """
        # Handle both List[int] and torch.Tensor inputs
        if isinstance(tokens, torch.Tensor):
            token_list = [int(t.item()) for t in tokens.squeeze() if t not in self.SPECIAL_TOKENS.values()]
        else:
            token_list = [int(t) for t in tokens if t not in self.SPECIAL_TOKENS.values()]

        token_bytes = b"".join([self.vocab.get(t, b"") for t in token_list])
        text = token_bytes.decode("utf-8", errors="replace")
        return text

    def train(self, text: str) -> None:
        """Trains the tokenizer on a given text to build the vocabulary and
        merge operations. This is the original in-memory training method.

        Args:
            text (str): The input text to train on.
        """
        # Use streaming training with the text wrapped as a single-item iterator
        self.train_streaming(iter([text]))

    def train_streaming(self, text_iterator: Iterator[str], show_progress: bool = True) -> None:
        """Trains the tokenizer on streaming text data using word-frequency BPE.

        This is memory-efficient: instead of building a giant token sequence,
        we maintain a Counter of (word -> frequency) where each word is a tuple
        of token ids. This is the standard approach used by GPT-2/tiktoken.

        Args:
            text_iterator: Iterator yielding text chunks/lines.
            show_progress: Whether to show progress bar.
        """
        # Step 1: Build word frequencies (pre-tokenize and count)
        # word_freqs: {tuple of bytes: count}
        word_freqs: Counter = Counter()

        print("Building word frequencies from text...")
        for text_chunk in text_iterator:
            # Pre-tokenize using GPT-2 regex
            pieces = re.findall(self._GPT2_PAT, text_chunk)
            for piece in pieces:
                # Convert to tuple of bytes (each byte is an int)
                word = tuple(piece.encode("utf-8"))
                word_freqs[word] += 1

        print(f"Found {len(word_freqs)} unique pre-tokens")

        # Step 2: BPE training loop
        # word_freqs now contains {tuple[int, ...]: frequency}
        # We'll iteratively merge the most frequent pair

        num_merges = self.vocab_size - self.BASE_VOCAB_SIZE - len(self.SPECIAL_TOKENS)

        iterator = range(num_merges)
        if show_progress:
            iterator = tqdm(iterator, desc="Training BPE")

        for merge_idx in iterator:
            # Count all pairs weighted by word frequency
            pair_counts: Counter = Counter()
            for word, freq in word_freqs.items():
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                print(f"No more pairs to merge after {merge_idx} merges")
                break

            # Find most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            new_token_id = self.BASE_VOCAB_SIZE + merge_idx + len(self.SPECIAL_TOKENS)
            self.merges[best_pair] = new_token_id

            # Apply merge to all words in word_freqs
            new_word_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                new_word = self._apply_merge(word, best_pair, new_token_id)
                new_word_freqs[new_word] += freq

            word_freqs = new_word_freqs

        # Build vocabulary from merges
        print("Building vocabulary...")
        for (a, b), idx in tqdm(self.merges.items(), desc="Building vocab"):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

        # Build merge ranks for fast encoding
        self._build_merge_ranks()

    def _apply_merge(self, word: Tuple[int, ...], pair: Tuple[int, int], new_token: int) -> Tuple[int, ...]:
        """Apply a merge operation to a word (tuple of token ids).

        Args:
            word: Tuple of token ids
            pair: The pair to merge
            new_token: The new token id for the merged pair

        Returns:
            New word tuple with merges applied
        """
        if len(word) < 2:
            return word

        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def save(self, directory: str) -> None:
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
    def load(cls, directory: str) -> "Tokenizer":
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
        tokenizer._build_merge_ranks()  # Build ranks for fast encoding
        return tokenizer

    def clear_cache(self) -> None:
        """Clear the encoding cache. Useful if memory is tight."""
        if self._encode_piece_cached is not None:
            self._encode_piece_cached.cache_clear()
