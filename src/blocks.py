import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_legacy_buffers(state_dict, prefix, names):
    """Discard derived buffers that checkpoints saved before they became non-persistent still carry."""
    for name in names:
        state_dict.pop(prefix + name, None)


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


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoFormer, arXiv:2104.09864) over adjacent feature pairs.

    Dtype policy: rotation angles and the cos/sin caches are always FP32. They are cast to the
    input dtype only for the final rotation, so the output dtype equals the input dtype (FP32,
    FP16 or BF16) and autocast-produced half-precision Q/K stay half precision.

    The caches are non-persistent and rebuilt on demand when a request needs more positions,
    the input lives on another device, or a module-wide ``.half()``/``.to(dtype)`` downcast them.
    """

    LEGACY_BUFFERS = ("inv_freq", "cos_cached", "sin_cached")

    def __init__(self, dim, max_seq_len, base=10000.0):
        """
        Parameters:
            dim: rotary feature dimension; must be a positive even integer
            max_seq_len: number of positions to precompute; longer requests grow the cache
            base: frequency base of the rotation angles
        """
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be a positive even integer, got {dim}")
        if max_seq_len <= 0:
            raise ValueError(f"RoPE max_seq_len must be positive, got {max_seq_len}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.register_buffer("cos_cached", torch.empty(0, dim // 2), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0, dim // 2), persistent=False)
        self._build_cache(max_seq_len, torch.device("cpu"))

    def _angles(self, positions):
        """FP32 rotation angles of shape [*positions.shape, dim/2], on the positions' device."""
        exponents = torch.arange(0, self.dim, 2, device=positions.device, dtype=torch.float32) / self.dim
        inv_freq = 1.0 / (self.base**exponents)
        return positions.to(torch.float32).unsqueeze(-1) * inv_freq

    def _build_cache(self, length, device):
        # A cache first built under inference_mode would otherwise break later training forwards.
        with torch.inference_mode(False):
            angles = self._angles(torch.arange(length, device=device))
            self.cos_cached = torch.cos(angles)
            self.sin_cached = torch.sin(angles)

    def _cached_cos_sin(self, start, length, device):
        end = start + length
        cache = self.cos_cached
        cache_len = cache.size(0)
        if cache_len < end:
            # Grow geometrically so token-by-token decoding past the cache does not rebuild every step.
            self._build_cache(max(end, 2 * cache_len), device)
        elif cache.device != device or cache.dtype != torch.float32:
            self._build_cache(cache_len, device)
        return self.cos_cached[start:end], self.sin_cached[start:end]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        drop_legacy_buffers(state_dict, prefix, self.LEGACY_BUFFERS)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x, position_ids=None, offset=0):
        """
        Rotate feature pairs of ``x`` by their absolute positions.

        Args:
            x: floating tensor [batch, seq_len, dim] or [batch, n_heads, seq_len, dim]
            position_ids: optional integer positions [seq_len] or [batch, seq_len]
            offset: position of the first token when ``position_ids`` is omitted,
                e.g. the number of tokens already in a KV cache
        Returns:
            tensor of the same shape, dtype and device as ``x``
        """
        if x.dim() not in (3, 4):
            raise ValueError(f"RoPE expects a 3D or 4D tensor, got shape {tuple(x.shape)}")
        if x.size(-1) != self.dim:
            raise ValueError(f"RoPE expects last dimension {self.dim}, got {x.size(-1)}")
        if not x.is_floating_point():
            raise TypeError(f"RoPE expects a floating tensor, got {x.dtype}")
        seq_len = x.size(-2)

        if position_ids is None:
            if offset < 0:
                raise ValueError(f"RoPE offset must be non-negative, got {offset}")
            cos, sin = self._cached_cos_sin(offset, seq_len, x.device)  # [seq_len, dim/2]
        else:
            if offset != 0:
                raise ValueError("Pass either position_ids or offset to RoPE, not both")
            if position_ids.dtype.is_floating_point or position_ids.dtype.is_complex:
                raise TypeError(f"RoPE position_ids must be integers, got {position_ids.dtype}")
            if position_ids.dim() not in (1, 2) or position_ids.size(-1) != seq_len:
                raise ValueError(
                    f"RoPE position_ids must have shape [{seq_len}] or [batch, {seq_len}], "
                    f"got {tuple(position_ids.shape)}"
                )
            angles = self._angles(position_ids.to(x.device))
            cos, sin = torch.cos(angles), torch.sin(angles)  # [(batch,) seq_len, dim/2]
            if x.dim() == 4 and position_ids.dim() == 2:
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # broadcast across heads

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        even, odd = x.reshape(*x.shape[:-1], self.dim // 2, 2).unbind(-1)
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class PositionalEncoding(nn.Module):
    """Additive sinusoidal encoding from "Attention Is All You Need" (arXiv:1706.03762, section 3.5).

    PE[pos, 2i] = sin(pos / base^(2i/d)) and PE[pos, 2i+1] = cos(pos / base^(2i/d)): each sine/cosine
    pair shares one frequency. The table is derived, so it is a non-persistent buffer.
    """

    LEGACY_BUFFERS = ("pos_enc",)

    def __init__(self, max_seq_len, embed_dim, base=10000.0):
        """
        Parameters:
            max_seq_len: longest sequence accepted; longer inputs raise ValueError
            embed_dim: dimension of word embedding; must be a positive even integer
            base: frequency base of the encoding
        """
        super().__init__()
        if embed_dim <= 0 or embed_dim % 2 != 0:
            raise ValueError(f"Sinusoidal embed_dim must be a positive even integer, got {embed_dim}")
        if max_seq_len <= 0:
            raise ValueError(f"Sinusoidal max_seq_len must be positive, got {max_seq_len}")
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim))
        angles = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1) * inv_freq
        pos_enc = torch.zeros(max_seq_len, embed_dim)
        pos_enc[:, 0::2] = torch.sin(angles)
        pos_enc[:, 1::2] = torch.cos(angles)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0), persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        drop_legacy_buffers(state_dict, prefix, self.LEGACY_BUFFERS)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: embeddings [batch, seq_len, embed_dim]
        Returns:
            x plus the encoding of positions 0..seq_len-1, in x's dtype
        """
        if x.size(1) > self.max_seq_len:
            raise ValueError(f"Sequence length {x.size(1)} exceeds sinusoidal max_seq_len {self.max_seq_len}")
        return x + self.pos_enc[:, : x.size(1)].to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, pos_encoding_type, max_seq_len=512, dropout_rate=0.2):
        """
        Parameters:
            embed_dim: dimension of embeddings
            n_heads: number of attention heads
            pos_encoding_type: type of positional encoding ("sinusoidal" or "rotary")
            max_seq_len: maximum sequence length
            dropout_rate: dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.do_0 = nn.Dropout(dropout_rate)
        self.do_1 = nn.Dropout(dropout_rate)
        self.pos_encoding_type = pos_encoding_type

        if pos_encoding_type == "rotary":
            self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)

        assert self.head_dim * n_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None, is_causal=False):
        """Attend to keys/values using an optional boolean blocked-position mask.

        True entries are excluded from attention. The mask must broadcast to
        (batch, n_heads, query_length, key_length); causal and padding masks are
        constructed separately by callers.
        """
        if mask is not None and mask.dtype != torch.bool:
            raise TypeError("Attention mask must be boolean, with True marking blocked positions")
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape to (batch_size, n_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if selected
        if self.pos_encoding_type == "rotary":
            Q = self.rope(Q)
            K = self.rope(K)

        # This API uses True for blocked positions; PyTorch SDPA uses True for allowed positions.
        allowed_mask = None if mask is None else ~mask
        attn_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=allowed_mask,
            dropout_p=self.do_0.p if self.training else 0.0,
            is_causal=is_causal,
        )
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.do_1(self.out_linear(attn_output))

        return output


class GPTDecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        expansion_factor=4,
        n_heads=8,
        pos_encoding_type="rotary",
        dropout_rate=0.2,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embed_dim,
            n_heads,
            pos_encoding_type=pos_encoding_type,
            dropout_rate=dropout_rate,
        )
        self.norm_0 = nn.LayerNorm(embed_dim)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x, mask):
        att = self.attention(x, x, x, mask, is_causal=mask is None)
        add_and_norm = self.norm_0(att + x)
        ffn_out = self.ffn(add_and_norm)
        out = self.norm_1(ffn_out + add_and_norm)
        return out


class GPTDecoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embed_dim,
        num_layers=2,
        expansion_factor=4,
        n_heads=8,
        dropout_rate=0.2,
        pos_encoding_type="rotary",
    ):
        super().__init__()

        self.word_embedding = Embedding(target_vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList(
            [
                GPTDecoderBlock(embed_dim, expansion_factor, n_heads, pos_encoding_type, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.fully_connected = nn.Linear(embed_dim, target_vocab_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        x = self.word_embedding(x)
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        out = self.fully_connected(x)
        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, pos_encoding_type="rotary", dropout_rate=0.2):
        """
        Parameters:
            embed_dim: dimension of the embedding vector
            expansion_factor: factor which determines the dimension of the linear layer
            n_heads: number of attention heads.
        """
        super().__init__()

        self.attention = MultiHeadAttention(
            embed_dim, n_heads, pos_encoding_type=pos_encoding_type, dropout_rate=dropout_rate
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.GELU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, key, query, value):
        attention = self.attention(query=query, key=key, value=value)  # 32x10x512

        norm1_out = self.dropout1(self.norm1(attention + query))  # 32x10x512
        ff_out = self.feed_forward(norm1_out)  # 32x10x512 -> #32x10x2048 -> 32x10x512

        norm2_out = self.dropout2(self.norm2(ff_out + norm1_out))  # 32x10x512

        return norm2_out


class TransformerEncoder(nn.Module):
    def __init__(
        self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8, pos_encoding_type="rotary"
    ):
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
            [
                TransformerEncoderBlock(embed_dim, expansion_factor, n_heads, pos_encoding_type)
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.pos_encoding(embed_out)

        for layer in self.layers:
            out = layer(out, out, out)

        return out


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, pos_encoding_type="rotary", dropout_rate=0.2):
        super().__init__()

        self.attention = MultiHeadAttention(
            embed_dim, n_heads, pos_encoding_type=pos_encoding_type, dropout_rate=dropout_rate
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.do = nn.Dropout(dropout_rate)
        self.encoder_block = TransformerEncoderBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, x, value, mask):
        att = self.attention(x, x, x, mask)
        query = self.do(self.norm(att + x))
        out = self.encoder_block(key, query, value)
        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embed_dim,
        seq_len,
        num_layers=2,
        expansion_factor=4,
        n_heads=8,
        dropout_rate=0.2,
        pos_encoding_type="rotary",
    ):
        super().__init__()

        self.word_embedding = Embedding(target_vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(embed_dim, expansion_factor, n_heads, pos_encoding_type)
                for _ in range(num_layers)
            ]
        )

        self.fully_connected = nn.Linear(embed_dim, target_vocab_size)
        self.do = nn.Dropout(dropout_rate)

    def forward(self, x, enc_out, mask):
        x = self.word_embedding(x)
        x = self.pos_enc(x)
        x = self.do(x)

        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask)

        out = F.softmax(self.fully_connected(x))
        return out
