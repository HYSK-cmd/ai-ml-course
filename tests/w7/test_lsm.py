"""Week 7 autograder: the LSM-tree KV store."""
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from sysdesign.lsm import LSMStore

ROOT = Path(__file__).resolve().parents[2]
WEIRD = "tab\there\nnewline, unicode 한국어 🚀, quote\" and 'single' \\ backslash"


def test_crud(tmp_path):
    s = LSMStore(str(tmp_path))
    assert s.get("a") is None
    s.put("a", "1")
    s.put("b", WEIRD)
    s.put("a", "2")
    assert s.get("a") == "2" and s.get("b") == WEIRD
    s.delete("a")
    assert s.get("a") is None
    s.delete("never-existed")
    s.close()


def test_flush_and_reads_across_sstables(tmp_path):
    s = LSMStore(str(tmp_path), memtable_limit=100, compaction_threshold=100)
    for i in range(1000):
        s.put(f"k{i:04d}", f"v{i}")
    assert s.sstable_count >= 9
    assert list(tmp_path.glob("*.sst")), "SSTables are *.sst files in the directory"
    for i in range(0, 1000, 37):
        assert s.get(f"k{i:04d}") == f"v{i}"
    s.put("k0005", "new")  # newer value shadows the one in an old SSTable
    s.delete("k0006")      # tombstone shadows an old SSTable value
    s.flush()
    assert s.get("k0005") == "new" and s.get("k0006") is None
    s.close()


def test_scan(tmp_path):
    s = LSMStore(str(tmp_path), memtable_limit=7, compaction_threshold=100)
    for i in range(50):
        s.put(f"{i:02d}", str(i))
    s.delete("10")
    s.put("11", "eleven")
    assert s.scan("08", "13") == [("08", "8"), ("09", "9"), ("11", "eleven"), ("12", "12")]
    assert len(s.scan()) == 49
    assert s.scan(end="02") == [("00", "0"), ("01", "1")]
    s.close()


def test_persistence_across_reopen(tmp_path):
    s = LSMStore(str(tmp_path), memtable_limit=50)
    for i in range(120):
        s.put(str(i), str(i * i))
    s.delete("7")
    s.close()
    s2 = LSMStore(str(tmp_path), memtable_limit=50)
    assert s2.get("11") == "121" and s2.get("7") is None and s2.get("119") == str(119 * 119)
    s2.close()


CRASHER = """
import os, sys
sys.path.insert(0, {root!r})
from sysdesign.lsm import LSMStore
s = LSMStore({d!r}, memtable_limit=1000)
for i in range(300):
    s.put(f"key{{i}}", f"val{{i}}")
s.delete("key5")
os._exit(0)  # crash: no close(), no flush of the memtable
"""


def crash_writer(d):
    code = CRASHER.format(root=str(ROOT), d=str(d))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_crash_recovery_from_wal(tmp_path):
    crash_writer(tmp_path)
    assert (tmp_path / "wal.log").exists(), "the WAL must be <directory>/wal.log"
    s = LSMStore(str(tmp_path))
    assert s.get("key0") == "val0" and s.get("key299") == "val299" and s.get("key5") is None
    s.close()


def test_torn_wal_tail_is_ignored(tmp_path):
    crash_writer(tmp_path)
    with open(tmp_path / "wal.log", "ab") as f:
        f.write(b"\x00\x00\x01\x00garbage-from-a-half-written-record")
    s = LSMStore(str(tmp_path))
    assert s.get("key298") == "val298" and s.get("key5") is None
    s.put("after", "recovery")  # must keep working
    s.close()
    s2 = LSMStore(str(tmp_path))
    assert s2.get("after") == "recovery" and s2.get("key1") == "val1"
    s2.close()


def test_compaction_merges_and_drops_tombstones(tmp_path):
    s = LSMStore(str(tmp_path), memtable_limit=200, compaction_threshold=100)
    for i in range(2000):
        s.put(f"k{i}", "x" * 50)
    for i in range(1900):
        s.delete(f"k{i}")
    s.flush()
    size_before = sum(p.stat().st_size for p in tmp_path.glob("*.sst"))
    s.compact()
    assert s.sstable_count == 1
    size_after = sum(p.stat().st_size for p in tmp_path.glob("*.sst"))
    assert size_after < size_before / 5, "compaction must drop deleted and shadowed data"
    assert s.get("k1950") == "x" * 50 and s.get("k10") is None
    assert len(s.scan()) == 100
    s.close()


def test_automatic_compaction(tmp_path):
    s = LSMStore(str(tmp_path), memtable_limit=10, compaction_threshold=3)
    for i in range(200):
        s.put(f"k{i % 25}", str(i))
    assert s.sstable_count < 3
    assert s.get("k7") == str(175 + 7)
    s.close()


def test_bloom_filters_skip_sstables(tmp_path):
    s = LSMStore(str(tmp_path), memtable_limit=500, compaction_threshold=100)
    for i in range(5000):
        s.put(f"present-{i}", "v")
    s.flush()
    n_tables = s.sstable_count
    before = s.stats["sstable_reads"]
    for i in range(1000):
        assert s.get(f"absent-{i}") is None
    reads = s.stats["sstable_reads"] - before
    assert reads < 0.05 * 1000 * n_tables, f"{reads} SSTable reads for 1000 missing keys over {n_tables} tables"
    s.close()


def test_random_ops_match_dict_oracle(tmp_path):
    rng = random.Random(7)
    oracle = {}
    s = LSMStore(str(tmp_path), memtable_limit=40, compaction_threshold=3)
    for step in range(4000):
        k = f"k{rng.randrange(300)}"
        op = rng.random()
        if op < 0.6:
            v = f"v{step}"
            s.put(k, v)
            oracle[k] = v
        elif op < 0.8:
            s.delete(k)
            oracle.pop(k, None)
        else:
            assert s.get(k) == oracle.get(k), f"step {step}: get({k})"
        if step % 997 == 0:  # reopen now and then
            s.close()
            s = LSMStore(str(tmp_path), memtable_limit=40, compaction_threshold=3)
    assert s.scan() == sorted(oracle.items())
    s.close()
