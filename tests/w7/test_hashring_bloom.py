"""Week 7 autograder: consistent hashing + Bloom filter."""
import math
import statistics
import time

import pytest

from sysdesign.bloom import BloomFilter
from sysdesign.hashring import HashRing

KEYS = [f"user:{i}" for i in range(20000)]


def test_empty_and_errors():
    r = HashRing()
    with pytest.raises(LookupError):
        r.get_node("x")
    r.add_node("a")
    with pytest.raises(ValueError):
        r.add_node("a")
    with pytest.raises(KeyError):
        r.remove_node("zzz")
    assert r.get_node("anything") == "a" and r.nodes == {"a"}


def test_deterministic():
    a = HashRing(["n1", "n2", "n3"])
    b = HashRing(["n3", "n1", "n2"])  # insertion order must not matter
    assert all(a.get_node(k) == b.get_node(k) for k in KEYS[:2000])


def test_adding_a_node_moves_about_1_over_n_and_only_to_it():
    r = HashRing([f"n{i}" for i in range(9)], vnodes=150)
    before = {k: r.get_node(k) for k in KEYS}
    r.add_node("n9")
    after = {k: r.get_node(k) for k in KEYS}
    moved = [k for k in KEYS if before[k] != after[k]]
    assert all(after[k] == "n9" for k in moved), "keys may only move TO the new node"
    frac = len(moved) / len(KEYS)
    assert 0.05 < frac < 0.15, f"expected ~1/10 of keys to move, got {frac:.3f}"


def test_removing_a_node_only_moves_its_keys():
    r = HashRing([f"n{i}" for i in range(6)])
    before = {k: r.get_node(k) for k in KEYS}
    r.remove_node("n3")
    for k in KEYS:
        if before[k] != "n3":
            assert r.get_node(k) == before[k]
        else:
            assert r.get_node(k) != "n3"


def test_virtual_nodes_balance_load():
    def spread(vnodes):
        r = HashRing([f"n{i}" for i in range(10)], vnodes=vnodes)
        counts = {n: 0 for n in r.nodes}
        for k in KEYS:
            counts[r.get_node(k)] += 1
        return statistics.pstdev(counts.values()) / statistics.mean(counts.values()), max(counts.values()) / statistics.mean(counts.values())
    cv1, _ = spread(1)
    cv200, peak200 = spread(200)
    assert cv200 < cv1 / 3, "more virtual nodes must smooth the distribution"
    assert peak200 < 1.25


def test_replica_placement():
    r = HashRing(["a", "b", "c", "d"])
    for k in KEYS[:500]:
        reps = r.get_nodes(k, 3)
        assert len(reps) == 3 and len(set(reps)) == 3 and reps[0] == r.get_node(k)
    assert sorted(r.get_nodes("k", 10)) == ["a", "b", "c", "d"]


def test_lookup_is_logarithmic():
    r = HashRing([f"n{i}" for i in range(50)], vnodes=200)  # 10k ring points
    t = time.perf_counter()
    for k in KEYS:
        r.get_node(k)
    assert time.perf_counter() - t < 1.0, "use bisect, not a linear scan"


# ------------------------------------------------------------------------------ bloom
def test_bloom_sizing_formulas():
    b = BloomFilter(10_000, 0.01)
    assert b.m == math.ceil(-10_000 * math.log(0.01) / math.log(2) ** 2)
    assert b.k == 7


def test_bloom_no_false_negatives_and_fp_rate():
    b = BloomFilter(20_000, 0.01)
    for i in range(20_000):
        b.add(f"in-{i}")
    assert all(f"in-{i}" in b for i in range(20_000))
    fp = sum(f"out-{i}" in b for i in range(50_000)) / 50_000
    assert fp < 0.02, f"false-positive rate {fp:.4f} for a 1% target"


def test_bloom_is_bit_packed_and_serializable():
    b = BloomFilter(10_000, 0.01)
    for i in range(100):
        b.add(str(i))
    data = b.to_bytes()
    assert len(data) < b.m // 8 + 64, "store bits, not bytes/bools"
    c = BloomFilter.from_bytes(data)
    assert c.m == b.m and c.k == b.k and all(str(i) in c for i in range(100))
