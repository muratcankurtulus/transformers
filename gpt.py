import warnings

import torch
import torch.nn as nn

from blocks import GPTDecoder

warnings.simplefilter("ignore")
print(torch.cuda.get_device_name(0))


class GPT(nn.Module):

    def __init__(self,
                 tgt_vocab_size,
                 embed_dim,
                 seq_len,
                 num_layers=2,
                 expansion_factor=4,
                 n_heads=8):
        super().__init__()

        self.decoder = GPTDecoder(tgt_vocab_size, embed_dim, seq_len,
                                  num_layers, expansion_factor, n_heads)

    def make_tgt_mask(self, tgt):
        bs, seq_len = tgt.shape
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return tgt_mask.to("cuda")

    def generate(self, input_ids, max_length):
        self.eval()
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        generated = input_ids.clone().to("cuda")
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(generated, self.make_tgt_mask(generated))
                next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)
        return generated

    def forward(self, x, mask):
        return self.decoder(x, mask)
