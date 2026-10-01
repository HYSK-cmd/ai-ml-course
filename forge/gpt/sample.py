from __future__ import annotations

import torch


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Return a copy of logits (B, V) with filtered-out entries set to -inf.
    top_k: keep the k largest per row (ties at the k-th value may all be kept).
    top_p (nucleus): keep the smallest set of highest-probability tokens whose cumulative softmax
      probability >= top_p. The top-1 token is always kept. Apply top_k first, then top_p."""
    raise NotImplementedError


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (B, V) -> next token ids (B,). Divide by temperature (> 0) first, then filter, softmax,
    torch.multinomial(..., generator=generator)."""
    raise NotImplementedError
