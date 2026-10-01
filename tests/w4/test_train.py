"""Week 4 autograder: training utilities + a real Shakespeare run (slow, GPU)."""
import math

import pytest
import torch

from forge.gpt.model import GPT, GPTConfig
from forge.gpt.tokenizer import CharTokenizer
from forge.gpt.train import TrainConfig, get_batch, load_checkpoint, lr_at, save_checkpoint, train
from tests.helpers import need_data


def test_get_batch():
    data = torch.arange(100)
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, block_size=8, batch_size=500, generator=g)
    assert x.shape == (500, 8) and y.shape == (500, 8)
    assert torch.equal(y[:, :-1], x[:, 1:]) and torch.equal(y[:, -1], x[:, -1] + 1)
    assert x.min() >= 0 and y.max() <= 99
    assert x[:, 0].max() >= 85, "windows should be able to start near the end"
    x2, _ = get_batch(data, 8, 500, generator=torch.Generator().manual_seed(0))
    assert torch.equal(x, x2)


def test_lr_schedule():
    kw = dict(warmup_steps=10, max_steps=110, max_lr=1e-3, min_lr=1e-4)
    assert math.isclose(lr_at(0, **kw), 1e-4)
    assert math.isclose(lr_at(9, **kw), 1e-3)
    assert math.isclose(lr_at(10, **kw), 1e-3)
    assert math.isclose(lr_at(60, **kw), 5.5e-4)  # halfway through the cosine
    assert math.isclose(lr_at(110, **kw), 1e-4)
    assert math.isclose(lr_at(500, **kw), 1e-4)
    vals = [lr_at(s, **kw) for s in range(10, 111)]
    assert all(b <= a for a, b in zip(vals, vals[1:]))


def test_checkpoint_roundtrip(tmp_path):
    tok = CharTokenizer.from_text("to be or not to be")
    torch.manual_seed(0)
    m = GPT(GPTConfig(tok.vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=16)).eval()
    path = str(tmp_path / "sub" / "ck.pt")
    save_checkpoint(path, m, tok, step=7)
    m2, tok2 = load_checkpoint(path)
    assert not m2.training
    idx = torch.tensor([tok.encode("not to")])
    torch.testing.assert_close(m(idx)[0], m2(idx)[0])
    assert tok2.decode(tok2.encode("to be")) == "to be"


def test_train_tiny_cpu_end_to_end(tmp_path):
    data = tmp_path / "tiny.txt"
    data.write_text("the quick brown fox jumps over the lazy dog. " * 200, encoding="utf-8")
    cfg = TrainConfig(data_path=str(data), block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0,
                      batch_size=16, max_steps=150, lr=3e-3, warmup_steps=10, eval_interval=50, eval_iters=5,
                      device="cpu", out_dir=str(tmp_path / "ck"))
    out = train(cfg)
    assert set(out) >= {"train_loss", "val_loss", "steps", "checkpoint"}
    assert out["val_loss"] < 1.0, "a periodic sentence should be easy to learn"
    m, tok = load_checkpoint(out["checkpoint"])
    sample = tok.decode(m.generate(torch.tensor([tok.encode("the quick")]), 20, top_k=1)[0].tolist())
    assert sample.startswith("the quick brown fox")


@pytest.mark.slow
def test_shakespeare_char_gpt():
    """~6.4 min on an RTX 3060 for the reference solution."""
    need_data("shakespeare.txt")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    out = train(TrainConfig(max_steps=2500, eval_interval=500, out_dir="checkpoints/gpt"))
    print(out)
    assert out["val_loss"] < 1.60
