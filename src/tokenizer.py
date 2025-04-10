import os
from typing import Dict, List, Tuple, Union

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

    def __init__(self, vocab_size: int = 1024):
        """Initializes the Tokenizer with a given vocabulary size.

        Args:
            vocab_size (int): The size of the vocabulary. Default is 1024.
        """
        self.vocab_size = vocab_size
        self.vocab = {idx: bytes([idx]) for idx in range(self.BASE_VOCAB_SIZE)}
        self.vocab.update({v: k.encode("utf-8") for k, v in self.SPECIAL_TOKENS.items()})
        self.merges = {}
        self._stats_cache = {}

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

    def encode(self, text: str) -> List[int]:
        """Encodes a given text into a list of token IDs using a more efficient algorithm.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: The list of token IDs.
        """
        # Encoding text to bytes first
        byte_tokens = list(text.encode("utf-8"))

        # Use a more efficient data structure - a linked list effectively
        # Store the tokens in a list that we won't resize, and use indices to navigate
        tokens = byte_tokens.copy()

        # For each position, store the next valid position
        next_indices = list(range(1, len(tokens) + 1))
        next_indices[-1] = -1  # End marker

        # Keep track of valid token positions
        valid_positions = list(range(len(tokens)))

        # Continue until no more merges are possible
        while len(valid_positions) >= 2:
            did_merge = False

            # Process each position
            i = 0
            while i != -1 and next_indices[i] != -1:
                next_i = next_indices[i]

                # Check if this pair can be merged
                if (tokens[i], tokens[next_i]) in self.merges:
                    # Get the merged token ID
                    merged_id = self.merges[(tokens[i], tokens[next_i])]

                    # Update the token at position i
                    tokens[i] = merged_id

                    # Update the next indices to skip the merged token
                    next_indices[i] = next_indices[next_i]

                    # Remove the merged position from valid positions
                    if next_i in valid_positions:
                        valid_positions.remove(next_i)

                    did_merge = True

                    # Don't advance i, as we need to check if the new pair can be merged
                else:
                    # Move to the next position
                    i = next_i

            # If no merges were performed, we're done
            if not did_merge:
                break

        # Collect the final tokens
        result = []
        i = 0
        while i != -1:
            result.append(tokens[i])
            i = next_indices[i]

        return result

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

        token_bytes = b"".join([self.vocab[t] for t in token_list])
        text = token_bytes.decode("utf-8", errors="replace")
        return text

    def train(self, text: str) -> None:
        """Trains the tokenizer on a given text to build the vocabulary and
        merge operations.

        Args:
            text (str): The input text to train on.
        """
        # This regex pattern is based on GPT-2's BPE tokenization approach
        # It splits text into words, numbers, punctuation, and whitespace
        pat = re.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )
        list_of_text = re.findall(pat, text)
        bytestrings = [item.encode("utf-8") for item in list_of_text]
        text_bytes = b"".join(bytestrings)
        tokens = list(map(int, text_bytes))

        # Create a list of tokens with their positions
        token_sequence = []
        for i, token in enumerate(tokens):
            token_sequence.append((i, token))

        # Initialize pair counts and positions
        pair_counts = {}
        pair_positions = {}

        # Initial count of all pairs
        for i in range(len(token_sequence) - 1):
            pair = (token_sequence[i][1], token_sequence[i + 1][1])
            if pair not in pair_counts:
                pair_counts[pair] = 0
                pair_positions[pair] = []
            pair_counts[pair] += 1
            pair_positions[pair].append(i)

        num_merges = self.vocab_size - self.BASE_VOCAB_SIZE - len(self.SPECIAL_TOKENS)
        for i in tqdm(range(num_merges)):
            if not pair_counts:
                break

            # Find the most frequent pair
            pair = max(pair_counts, key=pair_counts.get)
            new_token_id = self.BASE_VOCAB_SIZE + i + len(self.SPECIAL_TOKENS)
            self.merges[pair] = new_token_id

            # Update the token sequence and pair counts
            positions = sorted(pair_positions[pair], reverse=True)
            for pos in positions:
                # Check if pos and pos + 1 are valid indices in the *current* token_sequence
                if pos + 1 >= len(token_sequence):
                    continue  # This position is no longer valid due to previous merges

                # Check if the pair at pos is still the one we intend to merge.
                # It might have changed due to a prior merge in this same loop iteration
                # affecting pos-1 or pos+1.
                if (token_sequence[pos][1], token_sequence[pos + 1][1]) != pair:
                    continue  # Pair mismatch, skip this position

                # Remove the old pair
                first_token = token_sequence[pos]
                second_token = token_sequence[pos + 1]

                # Create the new merged token
                token_sequence[pos] = (first_token[0], new_token_id)
                token_sequence.pop(pos + 1)

                # Update pair counts
                # 1. Remove affected pairs
                if pos > 0:
                    left_pair = (token_sequence[pos - 1][1], first_token[1])
                    if left_pair in pair_counts:
                        pair_counts[left_pair] -= 1
                        # Check if left_pair is still in pair_positions and if pos - 1 is in its list
                        if left_pair in pair_positions and (pos - 1) in pair_positions[left_pair]:
                            pair_positions[left_pair].remove(pos - 1)
                        if pair_counts[left_pair] == 0:
                            del pair_counts[left_pair]
                            if left_pair in pair_positions:  # Also check here before deleting
                                del pair_positions[left_pair]

                # The check for right_pair (pos + 1 < len(token_sequence)) already ensures index validity
                # But we need a similar check for removing the position 'pos' itself
                if pos + 1 < len(token_sequence):  # This check needs refinement related to `second_token`
                    # Re-calculate right_pair based on potentially updated token_sequence[pos+1]
                    # However, the original logic used second_token, which might be stale if pos+1 was modified
                    # Let's stick to the original logic's intent but add safety checks
                    original_right_pair = (second_token[1], token_sequence[pos + 1][1])  # Pair *before* merge at pos+1

                    # Check if the pair involving the *original* second token exists and needs updating
                    if original_right_pair in pair_counts:
                        pair_counts[original_right_pair] -= 1
                        # Check if the pair exists and the position 'pos' is in its list
                        if original_right_pair in pair_positions and pos in pair_positions[original_right_pair]:
                            pair_positions[original_right_pair].remove(pos)
                        if pair_counts[original_right_pair] == 0:
                            del pair_counts[original_right_pair]
                            if original_right_pair in pair_positions:  # Also check here before deleting
                                del pair_positions[original_right_pair]

                # 2. Add new pairs
                if pos > 0:
                    new_left_pair = (token_sequence[pos - 1][1], new_token_id)
                    if new_left_pair not in pair_counts:
                        pair_counts[new_left_pair] = 0
                        pair_positions[new_left_pair] = []
                    pair_counts[new_left_pair] += 1
                    pair_positions[new_left_pair].append(pos - 1)

                if pos < len(token_sequence) - 1:
                    new_right_pair = (new_token_id, token_sequence[pos + 1][1])
                    if new_right_pair not in pair_counts:
                        pair_counts[new_right_pair] = 0
                        pair_positions[new_right_pair] = []
                    pair_counts[new_right_pair] += 1
                    pair_positions[new_right_pair].append(pos)

            # Remove the merged pair from counts
            del pair_counts[pair]
            del pair_positions[pair]

        # Build the vocabulary
        for (a, b), idx in tqdm(self.merges.items()):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

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
        return tokenizer
