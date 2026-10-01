"""Week 4 autograder: tokenizers."""
import time

import pytest

from forge.gpt.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer
from tests.helpers import need_data

TRICKY = "Hello, world! 안녕하세요 🤖 naïve café -- \n\t  spaces   and nbsp"


def test_char_tokenizer(tmp_path):
    t = CharTokenizer.from_text("hello world")
    assert t.vocab_size == 8
    assert t.encode(" dehlorw") == list(range(8)), "ids in sorted character order"
    assert t.decode(t.encode("hello")) == "hello"
    t.save(str(tmp_path / "c.json"))
    t2 = load_tokenizer(str(tmp_path / "c.json"))
    assert isinstance(t2, CharTokenizer) and t2.encode("world") == t.encode("world")


def test_untrained_bpe_is_bytes():
    t = BPETokenizer()
    assert t.vocab_size == 256
    assert t.encode("hé") == list("hé".encode("utf-8"))
    assert t.decode(t.encode(TRICKY)) == TRICKY


def test_bpe_wikipedia_example_with_tiebreak():
    t = BPETokenizer()
    t.train("aaabdaaabac", 259)
    assert t.merges == {(97, 97): 256, (256, 97): 257, (257, 98): 258}
    assert list(t.merges.values()) == [256, 257, 258], "merges must be kept in learned order"
    assert t.encode("aaabdaaabac") == [258, 100, 258, 97, 99]
    assert t.vocab_size == 259


def test_bpe_stops_when_no_repeats():
    t = BPETokenizer()
    t.train("abcdef", 300)
    assert t.vocab_size == 256


def test_bpe_encode_uses_merge_order_not_frequency():
    t = BPETokenizer()
    t.train("abababab cdcd", 258)  # learns (a,b)->256 first, then (256,256)->257
    assert t.merges[(97, 98)] == 256
    assert t.encode("abab") == [257]
    assert t.encode("xab") == [ord("x"), 256]


def test_bpe_roundtrip_and_save_load(tmp_path):
    (path,) = need_data("shakespeare.txt")
    text = path.read_text(encoding="utf-8")
    t = BPETokenizer()
    t0 = time.perf_counter()
    t.train(text[:100_000], 512)
    assert time.perf_counter() - t0 < 120
    assert t.vocab_size == 512
    held_out = text[100_000:200_000]
    ids = t.encode(held_out)
    assert t.decode(ids) == held_out
    assert t.decode(t.encode(TRICKY)) == TRICKY, "unseen unicode must round-trip"
    ratio = len(held_out.encode("utf-8")) / len(ids)
    assert ratio > 1.8, f"compression ratio {ratio:.2f} on held-out text"
    t.save(str(tmp_path / "b.json"))
    t2 = load_tokenizer(str(tmp_path / "b.json"))
    assert isinstance(t2, BPETokenizer) and t2.encode(held_out[:5000]) == ids[: len(t2.encode(held_out[:5000]))]
    assert t2.merges == t.merges


def test_decode_invalid_utf8_does_not_crash():
    t = BPETokenizer()
    assert "�" in t.decode([0xE2, 0x82])  # truncated 3-byte sequence
