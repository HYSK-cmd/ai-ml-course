"""Tokenizers. Pure Python -- no tokenizer libraries.

Both tokenizers share: ``encode(str) -> list[int]``, ``decode(list[int]) -> str``, ``vocab_size``,
``save(path)`` (JSON) and the module-level ``load_tokenizer(path)`` that returns the right class.
"""
from __future__ import annotations


class CharTokenizer:
    """One id per distinct character of the training text, ids assigned in sorted character order."""

    def __init__(self, chars: list[str] | None = None):
        raise NotImplementedError

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError


class BPETokenizer:
    """Byte-level BPE (GPT-2 style, without the regex pre-split -- that's a stretch goal).

    * Base vocabulary: the 256 byte values (id i <-> bytes([i])). An untrained tokenizer encodes
      text to its raw UTF-8 bytes.
    * ``train(text, vocab_size)``: repeatedly merge the most frequent adjacent pair into a new id
      (256, 257, ...). TIE-BREAK: among equally frequent pairs, pick the pair whose first occurrence
      in the current sequence is earliest. Stop at vocab_size (or when no pair occurs twice).
    * ``encode``: apply merges to new text in the order they were LEARNED (lowest new id first),
      not greedily by frequency.
    * ``decode``: concatenate bytes and decode UTF-8 with errors="replace".
    """

    def __init__(self) -> None:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def merges(self) -> dict[tuple[int, int], int]:
        """(a, b) -> new id, in the order learned."""
        raise NotImplementedError

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError


def load_tokenizer(path: str) -> CharTokenizer | BPETokenizer:
    raise NotImplementedError
