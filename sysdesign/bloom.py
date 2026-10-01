"""Bloom filter: 'definitely not present' or 'probably present'.

Size it from the target: m = ceil(-n ln p / (ln 2)^2) bits, k = max(1, round(m / n * ln 2)) hashes.
Use double hashing: h_i(x) = (h1 + i * h2) mod m with h1, h2 from one hashlib digest (e.g. the two
halves of sha256). Store bits in a bytearray (1 bit per slot, not 1 byte).
"""
from __future__ import annotations


class BloomFilter:
    def __init__(self, capacity: int, error_rate: float = 0.01):
        raise NotImplementedError

    @property
    def m(self) -> int:
        """Number of bits."""
        raise NotImplementedError

    @property
    def k(self) -> int:
        """Number of hash functions."""
        raise NotImplementedError

    def add(self, key: str) -> None:
        raise NotImplementedError

    def __contains__(self, key: str) -> bool:
        raise NotImplementedError

    def to_bytes(self) -> bytes:
        """Serialize (so SSTables can store their filter)."""
        raise NotImplementedError

    @classmethod
    def from_bytes(cls, data: bytes) -> BloomFilter:
        raise NotImplementedError
