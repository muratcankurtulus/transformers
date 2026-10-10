from typing import List, Union

import torch
import torch.nn as nn

from blocks import GPTDecoder


class GPT(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        embed_dim: int,
        num_layers: int = 2,
        expansion_factor: int = 4,
        n_heads: int = 8,
        pos_encoding_type="rotary",
        dropout_rate: float = 0.2,
    ):
        """
        Initialize the GPT model.

        Args:
            tgt_vocab_size (int): Target vocabulary size.
            embed_dim (int): Embedding dimension.
            num_layers (int, optional): Number of decoder layers. Defaults to 2.
            expansion_factor (int, optional): Expansion factor for feed-forward layers. Defaults to 4.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            pos_encoding_type (str, optional): Type of positional encoding. Defaults to "rotary".
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
        """
        super().__init__()

        self.decoder = GPTDecoder(
            tgt_vocab_size,
            embed_dim,
            num_layers,
            expansion_factor,
            n_heads,
            dropout_rate=dropout_rate,
            pos_encoding_type=pos_encoding_type,
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
        input_ids: Union[List[int], torch.Tensor],
        max_length: int,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        context_length: int = 256,
        forbidden_token_ids: List[int] | None = None,
    ) -> torch.Tensor:
        """
        Generate a sequence of tokens.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            max_length (int): Maximum length of the generated sequence.

        Returns:
            torch.Tensor: Generated sequence of token IDs.
        """
        if temperature < 0 or top_k < 0 or not 0 < top_p <= 1 or context_length < 1:
            raise ValueError("Invalid sampling settings")
        self.eval()
        device = next(self.parameters()).device
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        else:
            input_ids = input_ids.to(device)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("Generation requires a nonempty prompt")
        generated = input_ids.clone()
        for _ in range(max_length):
            context = generated[:, -context_length:]
            logits = self(context)[:, -1, :].float()
            if forbidden_token_ids:
                logits[:, forbidden_token_ids] = float("-inf")
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits /= temperature
                if top_k:
                    threshold = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if top_p < 1:
                    sorted_logits, sorted_indices = logits.sort(descending=True)
                    remove = sorted_logits.softmax(-1).cumsum(-1) > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    remove = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
                    logits = logits.masked_fill(remove, float("-inf"))
                next_token = torch.multinomial(logits.softmax(-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

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
