"""Shared autoregressive decoding: prompt validation, token selection and per-sequence EOS stopping."""

import math
import operator
from collections.abc import Callable, Sequence

import torch


def checked_int(value, name: str) -> int:
    """Return ``value`` as a Python int; NumPy and PyTorch integers are accepted, bools and floats are not."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None


def as_token_batch(token_ids, vocab_size: int, device: torch.device, name: str = "prompt") -> torch.Tensor:
    """Return a new nonempty [batch, length] long tensor of token IDs on ``device``.

    Accepts a list of IDs, a list of equal-length ID lists, or an integer tensor/array shaped [length] or
    [batch, length]. Variable-length batches are rejected because generation has no padding mask.
    """
    if isinstance(token_ids, torch.Tensor):
        tokens = token_ids
    else:
        try:
            tokens = torch.tensor(token_ids)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"{name} must be integer token IDs: a list, equal-length lists, or a tensor; "
                "variable-length batches need padding masks, which generation does not support"
            ) from error
    if tokens.numel() == 0:
        raise ValueError(f"{name} must contain at least one token per sequence, got shape {tuple(tokens.shape)}")
    if tokens.dtype == torch.bool or tokens.dtype.is_floating_point or tokens.dtype.is_complex:
        raise TypeError(f"{name} token IDs must be integers, got {tokens.dtype}")
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2:
        raise ValueError(f"{name} must have shape [length] or [batch, length], got {tuple(tokens.shape)}")
    low, high = int(tokens.min()), int(tokens.max())
    if low < 0 or high >= vocab_size:
        raise ValueError(f"{name} token IDs must be in [0, {vocab_size}), got values in [{low}, {high}]")
    # Copy so the returned history never aliases the caller's prompt.
    return tokens.to(device=device, dtype=torch.long, copy=True)


def validate_settings(
    vocab_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_token_id: int | None,
    forbidden_token_ids: Sequence[int] | None,
) -> None:
    """Reject settings that would crash mid-generation or make EOS stopping impossible."""
    if checked_int(max_new_tokens, "max_new_tokens") < 0:
        raise ValueError(f"max_new_tokens must be nonnegative, got {max_new_tokens}")
    if not (math.isfinite(temperature) and temperature >= 0):
        raise ValueError(f"temperature must be finite and nonnegative (0 selects greedy decoding), got {temperature}")
    if checked_int(top_k, "top_k") < 0:
        raise ValueError(f"top_k must be nonnegative (0 disables it), got {top_k}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    forbidden = set()
    if forbidden_token_ids is not None:
        forbidden = {checked_int(i, "forbidden_token_ids") for i in forbidden_token_ids}
    stop = set() if eos_token_id is None else {checked_int(eos_token_id, "eos_token_id")}
    for token_id in forbidden | stop:
        if not 0 <= token_id < vocab_size:
            raise ValueError(f"Token ID {token_id} is outside the vocabulary [0, {vocab_size})")
    if len(forbidden) == vocab_size:
        raise ValueError("forbidden_token_ids covers the whole vocabulary, leaving nothing to generate")
    if eos_token_id in forbidden:
        raise ValueError(f"eos_token_id {eos_token_id} is also forbidden, so generation could never stop on it")


def select_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    forbidden_token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Choose one token per row of [batch, vocab] logits and return [batch, 1]; temperature 0 is greedy.

    The input logits are not modified.
    """
    logits = logits.float()
    if forbidden_token_ids is not None and len(forbidden_token_ids):
        forbidden = torch.tensor(list(forbidden_token_ids), dtype=torch.long, device=logits.device)
        logits = logits.index_fill(-1, forbidden, float("-inf"))
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
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
    return torch.multinomial(logits.softmax(-1), num_samples=1)


def decode(
    next_logits: Callable[[torch.Tensor], torch.Tensor],
    tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    forbidden_token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Append up to ``max_new_tokens`` tokens to the [batch, length] history ``tokens``.

    ``next_logits`` maps the full history to [batch, vocab] logits for the next position. A row that
    emits ``eos_token_id`` is finished: its later positions are filled with ``eos_token_id`` and the
    loop ends as soon as every row has finished, so the result may be shorter than the full budget.
    """
    finished = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)
    for _ in range(max_new_tokens):
        next_token = select_next_token(next_logits(tokens), temperature, top_k, top_p, forbidden_token_ids)
        if eos_token_id is not None:
            next_token = next_token.masked_fill(finished.unsqueeze(1), eos_token_id)
            finished |= next_token.squeeze(1) == eos_token_id
        tokens = torch.cat((tokens, next_token), dim=1)
        if eos_token_id is not None and bool(finished.all()):
            break
    return tokens
