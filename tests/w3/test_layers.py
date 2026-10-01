"""Week 3 autograder: layers, conv/pool, losses -- forward values and gradients."""
import time

import numpy as np
import pytest

from forge.autograd import Tensor
from forge.nn import (BatchNorm1d, Conv2d, Flatten, LayerNorm, Linear, MaxPool2d, cross_entropy, mse_loss)
from tests.helpers import numgrad

rng = np.random.default_rng(0)


def grad_check_module(module, x, atol=1e-6):
    """Check d(sum(out * R))/d(input and every parameter) against finite differences."""
    module.to(np.float64)
    R = rng.normal(size=module(Tensor(x)).shape)
    params = module.parameters()

    def f(xv, *pv):
        saved = [p.data for p in params]
        for p, v in zip(params, pv):
            p.data = v
        out = float((module(Tensor(xv)).data * R).sum())
        for p, v in zip(params, saved):
            p.data = v
        return out
    xt = Tensor(x.copy(), requires_grad=True)
    module.zero_grad()
    (module(xt) * R).sum().backward()
    expected = numgrad(f, [x.copy()] + [p.data.copy() for p in params])
    np.testing.assert_allclose(xt.grad, expected[0], atol=atol, rtol=1e-4, err_msg="input grad")
    for p, e in zip(params, expected[1:]):
        np.testing.assert_allclose(p.grad, e, atol=atol, rtol=1e-4, err_msg="param grad")


# ---------------------------------------------------------------------- naive references (grader-owned)
def conv_ref(x, w, b, stride, pad):
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    N, C, H, W = x.shape
    O, _, k, _ = w.shape
    Ho, Wo = (H - k) // stride + 1, (W - k) // stride + 1
    out = np.zeros((N, O, Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            patch = x[:, :, i * stride:i * stride + k, j * stride:j * stride + k]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    return out + (b[None, :, None, None] if b is not None else 0)


def pool_ref(x, k, s):
    N, C, H, W = x.shape
    Ho, Wo = (H - k) // s + 1, (W - k) // s + 1
    out = np.zeros((N, C, Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            out[:, :, i, j] = x[:, :, i * s:i * s + k, j * s:j * s + k].max(axis=(2, 3))
    return out


# ---------------------------------------------------------------------- Linear
def test_linear_shapes_init_and_grads():
    lin = Linear(512, 256, rng=np.random.default_rng(0))
    assert lin.weight.shape == (256, 512) and lin.bias.shape == (256,)
    assert abs(lin.weight.data.std() - np.sqrt(2 / 512)) < 0.05 * np.sqrt(2 / 512), "He init"
    np.testing.assert_array_equal(lin.bias.data, 0)
    assert Linear(3, 2, bias=False).bias is None and len(Linear(3, 2, bias=False).parameters()) == 1
    small = Linear(4, 3, rng=np.random.default_rng(1))
    small.bias.data = rng.normal(size=3).astype(np.float32)
    x = rng.normal(size=(5, 4))
    np.testing.assert_allclose(small.to(np.float64)(Tensor(x)).data, x @ small.weight.data.T + small.bias.data)
    grad_check_module(small, x)


def test_linear_batched_3d_input():
    lin = Linear(4, 3).to(np.float64)
    x = rng.normal(size=(2, 5, 4))
    assert lin(Tensor(x)).shape == (2, 5, 3)
    grad_check_module(lin, x)


# ---------------------------------------------------------------------- normalization
def test_batchnorm_train_forward_and_running_stats():
    bn = BatchNorm1d(3, momentum=0.1).to(np.float64)
    x = rng.normal(2.0, 3.0, size=(32, 3))
    out = bn(Tensor(x)).data
    np.testing.assert_allclose(out, (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5), atol=1e-10)
    np.testing.assert_allclose(bn.running_mean, 0.1 * x.mean(0))
    np.testing.assert_allclose(bn.running_var, 0.9 + 0.1 * x.var(0, ddof=1))


def test_batchnorm_eval_uses_running_stats():
    bn = BatchNorm1d(2).to(np.float64)
    for _ in range(200):
        bn(Tensor(rng.normal(5.0, 2.0, size=(64, 2))))
    bn.eval()
    x = np.array([[5.0, 5.0], [7.0, 3.0]])
    out = bn(Tensor(x)).data  # a batch this small would be garbage with batch statistics
    np.testing.assert_allclose(out, (x - bn.running_mean) / np.sqrt(bn.running_var + 1e-5))
    np.testing.assert_allclose(bn.running_mean, 5.0, atol=0.3)


def test_batchnorm_gradients():
    bn = BatchNorm1d(3).to(np.float64)
    bn.weight.data = rng.normal(size=3)
    bn.bias.data = rng.normal(size=3)
    grad_check_module(bn, rng.normal(size=(6, 3)))


def test_layernorm_forward_and_gradients():
    ln = LayerNorm(5).to(np.float64)
    x = rng.normal(size=(2, 3, 5)) * 4 + 1
    out = ln(Tensor(x)).data
    np.testing.assert_allclose(out, (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5))
    ln.weight.data = rng.normal(size=5)
    grad_check_module(ln, x)


# ---------------------------------------------------------------------- conv & pool
@pytest.mark.parametrize("stride,pad,k", [(1, 0, 3), (1, 1, 3), (2, 1, 3), (2, 0, 2), (1, 2, 5)])
def test_conv_forward_matches_naive(stride, pad, k):
    conv = Conv2d(3, 4, k, stride=stride, padding=pad).to(np.float64)
    conv.bias.data = rng.normal(size=4)
    x = rng.normal(size=(2, 3, 9, 8))
    out = conv(Tensor(x))
    np.testing.assert_allclose(out.data, conv_ref(x, conv.weight.data, conv.bias.data, stride, pad), atol=1e-10)
    assert out.shape == (2, 4, (9 + 2 * pad - k) // stride + 1, (8 + 2 * pad - k) // stride + 1)


def test_conv_init():
    conv = Conv2d(16, 32, 3, rng=np.random.default_rng(0))
    assert conv.weight.shape == (32, 16, 3, 3)
    assert abs(conv.weight.data.std() - np.sqrt(2 / 144)) < 0.05 * np.sqrt(2 / 144)


@pytest.mark.parametrize("stride,pad", [(1, 1), (2, 1), (2, 0)])
def test_conv_gradients(stride, pad):
    conv = Conv2d(2, 3, 3, stride=stride, padding=pad).to(np.float64)
    grad_check_module(conv, rng.normal(size=(2, 2, 5, 6)))


@pytest.mark.parametrize("k,s", [(2, 2), (3, 2), (2, 1)])
def test_maxpool_forward_and_gradients(k, s):
    pool = MaxPool2d(k, s)
    x = rng.normal(size=(2, 3, 7, 6))
    np.testing.assert_allclose(pool(Tensor(x)).data, pool_ref(x, k, s))
    grad_check_module(pool, x)


def test_maxpool_ties_route_to_one_element():
    x = Tensor(np.ones((1, 1, 2, 2)), requires_grad=True)
    MaxPool2d(2)(x).sum().backward()
    assert x.grad.sum() == 1.0 and (x.grad == 1).sum() == 1


def test_flatten():
    x = Tensor(rng.normal(size=(4, 2, 3, 3)), requires_grad=True)
    y = Flatten()(x)
    assert y.shape == (4, 18)
    y.sum().backward()
    assert x.grad.shape == (4, 2, 3, 3)


def test_conv_speed():
    conv = Conv2d(16, 32, 3, padding=1)
    x = Tensor(rng.normal(size=(64, 16, 28, 28)).astype(np.float32), requires_grad=True)
    t = time.perf_counter()
    conv(x).sum().backward()
    elapsed = time.perf_counter() - t
    assert elapsed < 5.0, f"conv fwd+bwd took {elapsed:.2f}s -- too slow to train a CNN (Week 5 makes it faster)"


# ---------------------------------------------------------------------- losses
def test_cross_entropy_value_grad_and_stability():
    logits = rng.normal(size=(6, 4))
    y = np.array([0, 3, 1, 2, 2, 0])
    x = Tensor(logits, requires_grad=True)
    loss = cross_entropy(x, y)
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    assert np.isclose(loss.data, -np.log(p[np.arange(6), y]).mean())
    loss.backward()
    np.testing.assert_allclose(x.grad, (p - np.eye(4)[y]) / 6, atol=1e-12)
    big = cross_entropy(Tensor(np.array([[1e4, 0.0], [0.0, 1e4]])), np.array([0, 0]))
    assert np.isclose(big.data, 5e3)


def test_mse_loss():
    a = Tensor(rng.normal(size=(3, 2)), requires_grad=True)
    t = rng.normal(size=(3, 2))
    loss = mse_loss(a, t)
    assert np.isclose(loss.data, ((a.data - t) ** 2).mean())
    loss.backward()
    np.testing.assert_allclose(a.grad, 2 * (a.data - t) / 6)
