from collections.abc import Sequence

import torch
import torch.nn as nn

from blocks import GPTDecoder
from generation import as_token_batch, decode, validate_generation_settings


class GPT(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        embed_dim: int,
        num_layers: int = 2,
        expansion_factor: int = 4,
        n_heads: int = 8,
        pos_encoding_type: str = "rotary",
        dropout_rate: float = 0.2,
        max_seq_len: int = 512,
    ):
        """
        Initialize the GPT model.

        Args:
            tgt_vocab_size (int): Target vocabulary size.
            embed_dim (int): Embedding dimension.
            num_layers (int, optional): Number of decoder layers. Defaults to 2.
            expansion_factor (int, optional): Expansion factor for feed-forward layers. Defaults to 4.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            pos_encoding_type (str, optional): "rotary" rotates queries/keys; "sinusoidal" adds an
                absolute table to the embeddings. Defaults to "rotary".
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
            max_seq_len (int, optional): Trained context: sinusoidal table length, initial RoPE cache length,
                and the largest generation context window. Defaults to 512.
        """
        super().__init__()
        self.vocab_size = tgt_vocab_size
        self.max_seq_len = max_seq_len

        self.decoder = GPTDecoder(
            tgt_vocab_size,
            embed_dim,
            num_layers,
            expansion_factor,
            n_heads,
            dropout_rate=dropout_rate,
            pos_encoding_type=pos_encoding_type,
            max_seq_len=max_seq_len,
        )

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create a causal mask on the target device; True blocks a future position.

        The diagonal and past positions remain visible. This mask does not mask padding.

        Args:
            tgt (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Boolean mask of shape (seq_len, seq_len), shared by all batches and heads.
        """
        _, seq_len = tgt.shape
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device), diagonal=1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        context_length: int | None = None,
        forbidden_token_ids: Sequence[int] | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Continue each prompt autoregressively.

        Context limit: every step conditions on at most the last ``context_length`` tokens (a sliding
        window), so prompt plus output may exceed the trained context. The window itself may not
        exceed ``max_seq_len``, the context the positional encoding was built and trained for.

        Args:
            input_ids: Nonempty prompt as a list of IDs, equal-length ID lists, or an integer tensor
                shaped [length] or [batch, length].
            max_new_tokens: Upper bound on tokens appended to each prompt.
            temperature: Sampling temperature; 0 selects greedy decoding.
            top_k: Keep only the k most likely tokens; 0 disables the filter.
            top_p: Nucleus sampling probability mass in (0, 1].
            context_length: Sliding-window size; defaults to ``max_seq_len``.
            forbidden_token_ids: IDs that are never generated.
            eos_token_id: Stop a sequence once it emits this ID; finished rows are padded with it.

        Returns:
            torch.Tensor: [batch, prompt_length + n] token IDs with n <= max_new_tokens; n is smaller
            only when every sequence emitted ``eos_token_id``.
        """
        context_length = self.max_seq_len if context_length is None else context_length
        if not 1 <= context_length <= self.max_seq_len:
            raise ValueError(
                f"context_length must be in [1, max_seq_len={self.max_seq_len}], got {context_length}; "
                "positions past the trained context are unsupported"
            )
        validate_generation_settings(
            self.vocab_size, max_new_tokens, temperature, top_k, top_p, eos_token_id, forbidden_token_ids
        )
        tokens = as_token_batch(input_ids, self.vocab_size, next(self.parameters()).device)
        was_training = self.training
        self.eval()
        try:
            return decode(
                lambda history: self(history[:, -context_length:])[:, -1, :],
                tokens,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                eos_token_id,
                forbidden_token_ids,
            )
        finally:
            self.train(was_training)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the GPT model.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.decoder(x, mask)
