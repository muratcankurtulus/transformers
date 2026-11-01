from collections.abc import Sequence

import torch
import torch.nn as nn

from blocks import TransformerDecoder, TransformerEncoder
from generation import as_token_batch, decode, validate_settings


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        src_vocab_size,
        tgt_vocab_size,
        seq_len,
        num_layers=2,
        expansion_factor=4,
        n_heads=8,
        pos_encoding_type="rotary",
        dropout_rate=0.2,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.seq_len = seq_len
        self.encoder = TransformerEncoder(
            seq_len, src_vocab_size, embed_dim, num_layers, expansion_factor, n_heads, pos_encoding_type, dropout_rate
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, embed_dim, seq_len, num_layers, expansion_factor, n_heads, dropout_rate, pos_encoding_type
        )

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Return a boolean (batch, 1, length, length) causal mask on the target device.

        True blocks future positions; the diagonal and past positions remain visible.
        Padding is not included in this mask.
        """
        bs, tgt_len = tgt.shape
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device), diagonal=1)
        return tgt_mask.expand(bs, 1, tgt_len, tgt_len)

    @torch.no_grad()
    def generate(
        self,
        src: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
        tgt: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        forbidden_token_ids: Sequence[int] | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Decode target tokens for each source sequence.

        The source is encoded once. Every step feeds the full target history (start tokens plus all
        generated tokens) to the decoder with a causal mask rebuilt for that history's length.

        Context limit: there is no sliding window, because cropping would drop the decoder's start
        tokens. The last generated token is never fed back, so the decoder sees at most
        ``tgt_length + max_new_tokens - 1`` positions; requests above ``seq_len``, or sources longer
        than ``seq_len``, raise ValueError before any computation.

        Args:
            src: Nonempty source IDs, [length] or [batch, length].
            tgt: Nonempty decoder start IDs (e.g. BOS), [length] or [batch, length], same batch as ``src``.
            max_new_tokens: Upper bound on tokens appended to each target.
            temperature, top_k, top_p, forbidden_token_ids: As in ``GPT.generate``.
            eos_token_id: Stop a sequence once it emits this ID; finished rows are padded with it.

        Returns:
            torch.Tensor: [batch, tgt_length + n] target IDs with n <= max_new_tokens; n is smaller
            only when every sequence emitted ``eos_token_id``.
        """
        validate_settings(
            self.tgt_vocab_size, max_new_tokens, temperature, top_k, top_p, eos_token_id, forbidden_token_ids
        )
        device = next(self.parameters()).device
        src = as_token_batch(src, self.src_vocab_size, device, name="src")
        tgt = as_token_batch(tgt, self.tgt_vocab_size, device, name="tgt")
        if src.size(0) != tgt.size(0):
            raise ValueError(f"src and tgt batch sizes differ: {src.size(0)} != {tgt.size(0)}")
        if src.size(1) > self.seq_len:
            raise ValueError(f"src length {src.size(1)} exceeds seq_len {self.seq_len}")
        decoder_positions = tgt.size(1) + max_new_tokens - 1
        if decoder_positions > self.seq_len:
            raise ValueError(
                f"tgt length {tgt.size(1)} + max_new_tokens {max_new_tokens} needs {decoder_positions} "
                f"decoder positions, exceeding seq_len {self.seq_len}"
            )
        was_training = self.training
        self.eval()
        try:
            enc_out = self.encoder(src)
            return decode(
                lambda history: self.decoder(history, enc_out, self.make_tgt_mask(history))[:, -1, :],
                tgt,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                eos_token_id,
                forbidden_token_ids,
            )
        finally:
            self.train(was_training)

    def forward(self, src, tgt):
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src)

        out = self.decoder(tgt, enc_out, tgt_mask)
        return out
