"""Consistent hashing with virtual nodes (Dynamo, Cassandra, memcached clients).

* Hash function: int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "big").
* Each physical node owns ``vnodes`` points on the ring, at hash(f"{node}#{i}") for i in range(vnodes).
* get_node(key): the first ring point clockwise from hash(key) (wrap around). O(log n) with bisect.
* get_nodes(key, n): the first n DISTINCT physical nodes clockwise (replica placement).
* Adding a node must only move keys TO the new node; removing one only moves ITS keys.
"""
from __future__ import annotations

from collections.abc import Iterable


class HashRing:
    def __init__(self, nodes: Iterable[str] = (), vnodes: int = 100):
        raise NotImplementedError

    @property
    def nodes(self) -> set[str]:
        raise NotImplementedError

    def add_node(self, node: str) -> None:
        """ValueError if already present."""
        raise NotImplementedError

    def remove_node(self, node: str) -> None:
        """KeyError if absent."""
        raise NotImplementedError

    def get_node(self, key: str) -> str:
        """LookupError on an empty ring."""
        raise NotImplementedError

    def get_nodes(self, key: str, n: int) -> list[str]:
        """min(n, number of nodes) distinct nodes."""
        raise NotImplementedError
