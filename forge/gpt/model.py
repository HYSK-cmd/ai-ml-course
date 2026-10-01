"""A decoder-only transformer (GPT-2 architecture) with a KV cache.

Rules: you may use nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout, F.gelu, F.softmax,
F.cross_entropy. You may NOT use F.scaled_dot_product_attention, nn.MultiheadAttention or any
nn.Transformer* class -- attention is the point of this week. (A test greps this file.)

Architecture (param_count must match it):
  wte: Embedding(vocab_size, n_embd)   wpe: Embedding(block_size, n_embd)   (learned positions)
  n_layer x Block:  x = x + attn(ln_1(x));  x = x + mlp(ln_2(x))          (pre-LayerNorm)
     attn: c_attn Linear(C, 3C) -> split q,k,v -> n_head heads -> causal softmax(q k^T / sqrt(d)) v
           -> c_proj Linear(C, C)
     mlp:  c_fc Linear(C, 4C) -> GELU -> c_proj Linear(4C, C)
  ln_f: LayerNorm(C);  lm_head: Linear(C, vocab_size, bias=False) with weight TIED to wte.weight
  All Linear layers except lm_head have biases; dropout after attention probs, after each c_proj
  and after the embeddings.
Init: normal(0, 0.02) for Linear/Embedding weights, zeros for biases; every c_proj weight uses
  std 0.02 / sqrt(2 * n_layer) (why? -- residual stream variance; it's a quiz question).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


def param_count(cfg: GPTConfig) -> int:
    """Closed-form number of parameters of GPT(cfg) (tied lm_head counted once). Derive it by hand."""
    raise NotImplementedError


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, cache: Any = None) -> torch.Tensor:
        """x: (B, T, C). With a layer cache: append this call's k, v to it and attend over all
        cached positions (the T new queries still must not see each other's future)."""
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, cache: Any = None) -> torch.Tensor:
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx (B, T) int64, T <= block_size (raise ValueError otherwise).
        Returns (logits (B, T, vocab), loss or None). loss = mean cross-entropy vs targets (B, T)."""
        raise NotImplementedError

    def init_cache(self, batch_size: int) -> Any:
        """Empty KV cache for incremental decoding. Its structure is up to you (it must know how
        many positions it holds, for the positional embedding)."""
        raise NotImplementedError

    def step(self, idx: torch.Tensor, cache: Any) -> torch.Tensor:
        """Process NEW tokens idx (B, t) given everything already in ``cache``; update the cache.
        Returns logits (B, t, vocab) that must equal the corresponding slice of a full forward pass
        over (cached tokens + idx). Raise ValueError if the cache would exceed block_size."""
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None, use_cache: bool = True,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Autoregressively append max_new_tokens tokens to idx (B, T); return (B, T + max_new_tokens).
        Uses forge.gpt.sample.sample_next. Must work past block_size: without cache crop the context
        to the last block_size tokens; with cache, when it's full, rebuild it from the last
        block_size - 1 tokens (or fall back to no-cache)."""
        raise NotImplementedError
