import warnings

import torch
import torch.nn as nn

from blocks import TransformerDecoder, TransformerEncoder

warnings.simplefilter("ignore")
print(torch.cuda.get_device_name(0))


class Transformer(nn.Module):

    def __init__(self,
                 embed_dim,
                 src_vocab_size,
                 tgt_vocab_size,
                 seq_len,
                 num_layers=2,
                 expansion_factor=4,
                 n_heads=8):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.encoder = TransformerEncoder(seq_len, src_vocab_size, embed_dim,
                                          num_layers, expansion_factor,
                                          n_heads)
        self.decoder = TransformerDecoder(tgt_vocab_size, embed_dim, seq_len,
                                          num_layers, expansion_factor,
                                          n_heads)

    def make_tgt_mask(self, tgt):
        bs, tgt_len = tgt.shape
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len)).expand(
            bs, 1, tgt_len, tgt_len).to("cuda")
        return tgt_mask

    def generate(self, src, tgt):
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src)
        out_labels = []
        seq_len = src.shape[1]
        out = tgt

        for i in range(seq_len):
            out = self.decoder(out, enc_out, tgt_mask)
            out = out[:, -1, :]

            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, axis=0)

        return out_labels

    def forward(self, src, tgt):
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src)

        out = self.decoder(tgt, enc_out, tgt_mask)
        return out
