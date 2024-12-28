from typing import List, Union

import torch
import torch.nn as nn

from blocks import GPTDecoder


class GPT(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        embed_dim: int,
        seq_len: int,
        num_layers: int = 2,
        expansion_factor: int = 4,
        n_heads: int = 8,
        pos_encoding_type="rotary",
    ):
        """
        Initialize the GPT model.

        Args:
            tgt_vocab_size (int): Target vocabulary size.
            embed_dim (int): Embedding dimension.
            seq_len (int): Sequence length.
            num_layers (int, optional): Number of decoder layers. Defaults to 2.
            expansion_factor (int, optional): Expansion factor for feed-forward layers. Defaults to 4.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
        """
        super().__init__()

        self.decoder = GPTDecoder(
            tgt_vocab_size,
            embed_dim,
            seq_len,
            num_layers,
            expansion_factor,
            n_heads,
            pos_encoding_type=pos_encoding_type,
        )

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create a target mask for the decoder.

        Args:
            tgt (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Target mask tensor.
        """
        _, seq_len = tgt.shape
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return tgt_mask.to("cuda")

    def generate(self, input_ids: Union[List[int], torch.Tensor], max_length: int) -> torch.Tensor:
        """
        Generate a sequence of tokens.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            max_length (int): Maximum length of the generated sequence.

        Returns:
            torch.Tensor: Generated sequence of token IDs.
        """
        self.eval()
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0).to("cuda")
        else:
            input_ids = input_ids.to("cuda")

        generated = input_ids.clone().to("cuda")
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(generated, self.make_tgt_mask(generated))
                next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)
        return generated

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT model.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.decoder(x, mask)
