"""A log-structured merge-tree key-value store (the design behind LevelDB, RocksDB, Cassandra).

Write path:  put/delete -> append a record to the WAL (``<directory>/wal.log``) and flush() the file
             BEFORE touching the memtable -> update the in-memory memtable (a dict is fine; sort on flush).
             When the memtable holds >= memtable_limit keys, flush it to a new immutable SSTable
             (sorted key/value file, ``*.sst``) and truncate the WAL.
             After a flush, if there are >= compaction_threshold SSTables, compact() them.
Read path:   memtable first, then SSTables newest -> oldest. Deletes are TOMBSTONES (a delete must
             shadow older values in older SSTables).
SSTables:    each keeps an in-memory Bloom filter (your sysdesign.bloom) and a sparse index (every Nth
             key -> byte offset) so a get() touches at most one small block of one file per SSTable it
             can't rule out. Don't load whole SSTables into memory.
Recovery:    opening a directory loads the SSTables (keep their order -- e.g. numbered filenames) and
             replays the WAL. The WAL must tolerate a TORN TAIL: a crash mid-append can leave a partial or
             garbage record at the end -- detect it (length prefix + CRC32 per record) and ignore it and
             anything after it. Everything before it must be recovered.
compact():   merge ALL SSTables into one, keep only the newest version of each key, drop tombstones.
stats:       dict with "sstable_reads" = number of times a get() actually searched an SSTable file
             (a Bloom filter 'no' does NOT count).
Keys and values are arbitrary str (any unicode, including newlines and tabs).
"""
from __future__ import annotations


class LSMStore:
    def __init__(self, directory: str, memtable_limit: int = 1000, compaction_threshold: int = 4,
                 fsync: bool = False):
        raise NotImplementedError

    def put(self, key: str, value: str) -> None:
        raise NotImplementedError

    def get(self, key: str) -> str | None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def scan(self, start: str | None = None, end: str | None = None) -> list[tuple[str, str]]:
        """Live (key, value) pairs with start <= key < end (None = unbounded), sorted by key."""
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def compact(self) -> None:
        raise NotImplementedError

    @property
    def sstable_count(self) -> int:
        raise NotImplementedError

    @property
    def stats(self) -> dict[str, int]:
        raise NotImplementedError

    def close(self) -> None:
        """Flush the memtable and close files."""
        raise NotImplementedError
