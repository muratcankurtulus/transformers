import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Parameters:
            vocab_size: size of the vocabulary
            embed_dim: dimension of embeddings
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        Parameters:
            x: input vector
        Returns:
            embedding vector
        """
        return self.embed(x)


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_dim):
        """
        Class to calculate Positional Encoding
        Parameters:
            max_seq_len: maximum sequence length to expect
            emded_dim: dimension of word embedding
        """
        super().__init__()
        self.embed_dim = embed_dim

        # create a null matrix
        pos_enc = torch.zeros(max_seq_len, self.embed_dim)

        # apply the sine and cosine functions from 'attention is all you need' paper
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pos_enc[pos, i] = math.sin(pos / 10000 ** ((2 * i) / self.embed_dim))
                pos_enc[pos, i + 1] = math.cos(pos / 10000 ** ((2 * (i + 1)) / self.embed_dim))

        pos_enc = pos_enc.unsqueeze(0)

        # register buffer in Pytorch ->
        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """

        x = x + self.pos_enc[:, : x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.do = nn.Dropout(.2)

        assert self.head_dim * n_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape to (batch_size, n_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, n_heads, seq_len, seq_len)

        if mask is not None:
            scores.masked_fill_(mask, float("-inf"))  # Apply the mask

        attn_weights = F.softmax(
            scores, dim=-1)  # (batch_size, n_heads, seq_len, seq_len)
        attn_weights = self.do(attn_weights)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_linear(attn_output)

        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        """
        Parameters:
            embed_dim: dimension of the embedding vector
            expansion_factor: factor which determines the dimension of the linear layer
            n_heads: number of attention heads.
        """
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.GELU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value):
        attention = self.attention(key, query, value)  # 32x10x512

        norm1_out = self.dropout1(self.norm1(attention + query))  # 32x10x512
        ff_out = self.feed_forward(norm1_out)  # 32x10x512 -> #32x10x2048 -> 32x10x512

        norm2_out = self.dropout2(self.norm2(ff_out + norm1_out))  # 32x10x512

        return norm2_out


class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        """
        Parameters:
            seq_len: length of the sequence
            vocab_size: size of the vocabulary to the data
            embed_dim: dimension of embedding
            num_layers: number of encoder layers
            expansion_factor: factor which determines the linear layers in feed forward step
            n_heads: number of heads for multi head attention
        Returns:
            out: output of the encoder.
        """
        super().__init__()

        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)]
        )

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.pos_encoding(embed_out)

        for layer in self.layers:
            out = layer(out, out, out)

        return out


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.do = nn.Dropout(0.2)
        self.encoder_block = TransformerEncoderBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, x, value, mask):
        att = self.attention(x, x, x, mask)
        query = self.do(self.norm(att + x))
        out = self.encoder_block(key, query, value)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super().__init__()

        self.word_embedding = Embedding(target_vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(embed_dim, expansion_factor=4, n_heads=8) for _ in range(num_layers)]
        )

        self.fully_connected = nn.Linear(embed_dim, target_vocab_size)
        self.do = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        x = self.word_embedding(x)
        x = self.pos_enc(x)
        x = self.do(x)

        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask)

        out = F.softmax(self.fully_connected(x))
        return out


class GPTDecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm_0 = nn.LayerNorm(embed_dim)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.do_0 = nn.Dropout(0.2)
        self.do_1 = nn.Dropout(0.2)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.GELU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )

    def forward(self, x, mask):
        att = self.attention(x, x, x, mask)
        add_and_norm = self.do_0(self.norm_0(att + x))
        ffn_out = self.ffn(add_and_norm)
        out = self.do_1(self.norm_1(ffn_out + add_and_norm))
        return out


class GPTDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super().__init__()

        self.word_embedding = Embedding(target_vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(seq_len, embed_dim)
        self.layers = nn.ModuleList([GPTDecoderBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)])
        self.fully_connected = nn.Linear(embed_dim, target_vocab_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.do = nn.Dropout(.2)

    def forward(self, x, mask):
        x = self.word_embedding(x)
        x = self.pos_enc(x)
        x = self.do(x)

        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        out = self.fully_connected(x)
        return out
