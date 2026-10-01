"""The model side of the server. The HTTP layer (app.py) only knows the Engine protocol, so tests can
plug in a fake engine and you can swap models without touching the API."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = None


class Engine(Protocol):
    def generate_batch(self, reqs: list[GenerateRequest]) -> list[str]:
        """Continuations only (NOT including the prompt), one per request, same order."""
        ...

    def stream(self, req: GenerateRequest) -> Iterator[str]:
        """Yield the continuation piece by piece (one decoded token at a time)."""
        ...


class GPTEngine:
    """Wraps your Week 4 checkpoint.

    Core: generate_batch must return exactly what generating each request on its own would return
    (for greedy top_k=1). The simplest correct batching: group requests by (prompt length,
    max_new_tokens, temperature, top_k) and run each group as one batched generate() call.
    Stretch: left-pad different-length prompts and add an attention mask + per-row positions to
    your GPT so ANY requests share one forward pass (that's what vLLM & friends do, plus paging).
    """

    def __init__(self, ckpt_path: str, device: str = "cpu", seed: int = 0):
        raise NotImplementedError

    def generate_batch(self, reqs: list[GenerateRequest]) -> list[str]:
        raise NotImplementedError

    def stream(self, req: GenerateRequest) -> Iterator[str]:
        raise NotImplementedError
