"""Week 4 autograder: GPT model, attention correctness, KV cache, sampling."""
import inspect
import math

import pytest
import torch

import forge.gpt.model as model_module
from forge.gpt.model import GPT, GPTConfig, param_count
from forge.gpt.sample import filter_logits, sample_next

SMALL = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=4, n_embd=32)


def make(cfg=SMALL, seed=0):
    torch.manual_seed(seed)
    return GPT(cfg).eval()


def test_no_forbidden_apis():
    src = inspect.getsource(model_module)
    for bad in ["scaled_dot_product_attention", "MultiheadAttention", "nn.Transformer", "TransformerEncoder",
                "TransformerDecoder"]:
        assert bad not in src, f"write attention yourself -- found {bad}"


@pytest.mark.parametrize("cfg", [SMALL, GPTConfig(65), GPTConfig(50257, 1024, 12, 12, 768)])
def test_param_count_formula(cfg):
    if cfg.vocab_size == 50257:
        assert param_count(cfg) == 124_439_808  # GPT-2 small (no need to build it)
        return
    m = GPT(cfg)
    actual = sum(p.numel() for p in m.parameters())  # tied weights are counted once by .parameters()
    assert param_count(cfg) == actual
    if cfg.vocab_size == 65:
        assert actual == 10_770_816, "architecture differs from the spec in the model.py docstring"


def test_weight_tying_and_init():
    m = make(GPTConfig(vocab_size=100, block_size=16, n_layer=4, n_head=2, n_embd=64))
    emb = [p for p in m.parameters() if p.shape == (100, 64)]
    assert len(emb) == 1, "lm_head.weight must be the SAME tensor as the token embedding"
    stds = {n: p.std().item() for n, p in m.named_parameters() if p.dim() == 2}
    proj = [s for n, s in stds.items() if n.endswith("c_proj.weight")]
    assert proj and all(abs(s - 0.02 / math.sqrt(8)) < 0.003 for s in proj), "scaled init on c_proj"


def test_forward_shapes_and_initial_loss():
    m = make()
    idx = torch.randint(0, 50, (3, 20))
    logits, loss = m(idx)
    assert logits.shape == (3, 20, 50) and loss is None
    _, loss = m(idx, torch.randint(0, 50, (3, 20)))
    assert abs(loss.item() - math.log(50)) < 0.3, "at init the model should be ~uniform over the vocab"
    with pytest.raises(ValueError):
        m(torch.zeros(1, 33, dtype=torch.long))


def test_causal_mask_no_future_leak():
    m = make()
    idx = torch.randint(0, 50, (2, 32))
    a, _ = m(idx)
    idx2 = idx.clone()
    idx2[:, 20:] = torch.randint(0, 50, (2, 12))
    b, _ = m(idx2)
    torch.testing.assert_close(a[:, :20], b[:, :20])
    assert not torch.allclose(a[:, 20:], b[:, 20:])


def test_backward_reaches_every_parameter():
    m = make().train()
    idx = torch.randint(0, 50, (2, 16))
    _, loss = m(idx, idx)
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_kv_cache_matches_full_forward():
    m = make()
    idx = torch.randint(0, 50, (2, 30))
    full, _ = m(idx)
    cache = m.init_cache(2)
    parts = [m.step(idx[:, :10], cache)]  # prompt in one chunk
    parts += [m.step(idx[:, t:t + 1], cache) for t in range(10, 25)]  # then one token at a time
    parts.append(m.step(idx[:, 25:30], cache))  # then a chunk again
    torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError):
        m.step(idx[:, :3], cache)  # 30 + 3 > block_size 32


def test_generate_greedy_cache_equals_no_cache_and_past_block_size():
    m = make()
    prompt = torch.randint(0, 50, (2, 5))
    a = m.generate(prompt, 40, top_k=1, use_cache=True)  # 45 tokens > block_size 32
    b = m.generate(prompt, 40, top_k=1, use_cache=False)
    assert a.shape == (2, 45)
    assert torch.equal(a[:, :5], prompt)
    assert torch.equal(a[:, :32], b[:, :32]), "within block_size, cached and uncached greedy must agree"
    # greedy step check against an explicit argmax
    logits, _ = m(b[:, :20])
    assert torch.equal(b[:, 20], logits[:, -1].argmax(-1))


def test_generate_seeded_sampling_reproducible():
    m = make()
    p = torch.zeros(1, 1, dtype=torch.long)
    a = m.generate(p, 20, temperature=0.8, top_p=0.9, generator=torch.Generator().manual_seed(1))
    b = m.generate(p, 20, temperature=0.8, top_p=0.9, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


# ------------------------------------------------------------------ sampling utilities
def test_filter_top_k():
    x = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    out = filter_logits(x, top_k=2)
    assert torch.isinf(out[0, [0, 2, 4]]).all() and torch.equal(out[0, [1, 3]], x[0, [1, 3]])
    assert torch.equal(x, torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])), "must not modify the input"


def test_filter_top_p():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    x = probs.log()
    kept = ~torch.isinf(filter_logits(x, top_p=0.7))
    assert kept.tolist() == [[True, True, False, False]]  # 0.5 < 0.7 <= 0.8
    kept = ~torch.isinf(filter_logits(x, top_p=0.8))
    assert kept.tolist() == [[True, True, False, False]]  # cumulative 0.8 reaches p exactly
    kept = ~torch.isinf(filter_logits(x, top_p=0.1))
    assert kept.tolist() == [[True, False, False, False]]  # top-1 always kept
    shuffled = x[:, [2, 0, 3, 1]]
    kept = ~torch.isinf(filter_logits(shuffled, top_p=0.7))
    assert kept.tolist() == [[False, True, False, True]]


def test_sample_next_temperature_and_distribution():
    g = torch.Generator().manual_seed(0)
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]])).repeat(20000, 1)
    s = sample_next(logits, generator=g)
    freq = torch.bincount(s, minlength=3).float() / 20000
    torch.testing.assert_close(freq, torch.tensor([0.7, 0.2, 0.1]), atol=0.015, rtol=0)
    cold = sample_next(logits[:2000], temperature=0.05, generator=g)
    assert (cold == 0).float().mean() > 0.99
    assert (sample_next(logits[:100], top_k=1, generator=g) == 0).all()
