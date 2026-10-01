"""Week 1 autograder: forge.autograd.  Run: pytest tests/w1 -q"""
import numpy as np
import pytest

from forge.autograd import Tensor, no_grad, unbroadcast
from tests.helpers import numgrad

rng = np.random.default_rng(0)


def check(f, *shapes, positive=False, low=-2.0, high=2.0):
    """Compare autograd grads of scalar f against the grader's own finite differences."""
    arrays = [rng.uniform(0.5, 2.0, s) if positive else rng.uniform(low, high, s) for s in shapes]
    ts = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = f(*ts)
    assert out.data.size == 1, "test function must return a scalar"
    out.backward()
    expected = numgrad(lambda *xs: float(f(*[Tensor(x) for x in xs]).data), arrays)
    for t, e in zip(ts, expected):
        assert t.grad is not None, "leaf with requires_grad=True got no grad"
        assert t.grad.shape == t.data.shape, f"grad shape {t.grad.shape} != data shape {t.data.shape}"
        np.testing.assert_allclose(t.grad, e, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------- construction & basics
def test_dtype_rules():
    assert Tensor([1, 2, 3]).data.dtype == np.float64
    assert Tensor(np.ones(3, dtype=np.float32)).data.dtype == np.float32
    assert Tensor(np.arange(3)).data.dtype == np.float64
    t = Tensor(np.zeros((2, 3)))
    assert t.shape == (2, 3) and t.ndim == 2 and t.grad is None


def test_scalar_chain():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(-3.0, requires_grad=True)
    c = a * b + a ** 2 - b / a  # dc/da = b + 2a + b/a^2 ; dc/db = a - 1/a
    c.backward()
    assert np.isclose(a.grad, -3 + 4 + (-3) / 4)
    assert np.isclose(b.grad, 2 - 0.5)


def test_mixed_with_python_numbers_and_arrays():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    y = (2 * x + 1 - x / 2 + np.array([1.0, 1.0]) * x).sum()
    r = (3 - x).sum() + (1 / x).sum()
    (y + r).backward()
    np.testing.assert_allclose(x.grad, 2 - 0.5 + 1 - 1 - 1 / x.data ** 2)


def test_node_reused_accumulates():
    x = Tensor(3.0, requires_grad=True)
    y = x * x + x  # dy/dx = 2x + 1
    y.backward()
    assert np.isclose(x.grad, 7.0)


def test_diamond_graph():
    x = Tensor(np.array([0.5, -1.0]), requires_grad=True)
    a = x.tanh()
    b = a * 2
    c = a * a
    (b + c).sum().backward()
    t = np.tanh(x.data)
    np.testing.assert_allclose(x.grad, (2 + 2 * t) * (1 - t ** 2))


def test_grads_accumulate_across_backward_calls():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    (x * 3).sum().backward()
    (x * 3).sum().backward()
    np.testing.assert_allclose(x.grad, [6.0, 6.0])


def test_leaf_without_requires_grad_gets_none():
    x = Tensor(np.ones(3), requires_grad=True)
    c = Tensor(np.ones(3) * 2)  # constant
    (x * c).sum().backward()
    assert c.grad is None
    assert not c.requires_grad
    assert (x * c).requires_grad


def test_nonscalar_backward_needs_grad():
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    y = x * 2
    with pytest.raises(RuntimeError):
        y.backward()
    y.backward(np.ones((2, 2)))
    np.testing.assert_allclose(x.grad, 2 * np.ones((2, 2)))


def test_deep_graph_no_recursion_error():
    x = Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + x * 0.001
    y.backward()
    assert np.isclose(x.grad, 1 + 5000 * 0.001)


def test_no_grad():
    x = Tensor(np.ones(3), requires_grad=True)
    with no_grad():
        y = x * 2
        with no_grad():
            pass
        z = x * 2  # still disabled after the inner block exits
    assert not y.requires_grad and not z.requires_grad
    assert (x * 2).requires_grad, "grad mode must be restored after the block"
    with pytest.raises(ValueError):
        with no_grad():
            raise ValueError
    assert (x * 2).requires_grad, "grad mode must be restored even when the body raises"


def test_detach():
    x = Tensor(np.ones(2), requires_grad=True)
    d = (x * 2).detach()
    assert not d.requires_grad
    (d * x).sum().backward()
    np.testing.assert_allclose(x.grad, [2.0, 2.0])


# ---------------------------------------------------------------- unbroadcast
@pytest.mark.parametrize("gshape,shape", [
    ((3, 4), (4,)), ((3, 4), (3, 1)), ((3, 4), (1, 4)), ((2, 3, 4), (1, 4)),
    ((2, 3, 4), (3, 1)), ((2, 3, 4), ()), ((2, 3, 4), (2, 3, 4)), ((5, 1, 4), (1, 1, 4)),
])
def test_unbroadcast(gshape, shape):
    g = rng.normal(size=gshape)
    out = unbroadcast(g, shape)
    assert out.shape == shape
    # the defining property: <g, broadcast(v)> == <unbroadcast(g), v> for all v
    v = rng.normal(size=shape)
    assert np.isclose((g * np.broadcast_to(v, gshape)).sum(), (out * v).sum())


# ---------------------------------------------------------------- per-op gradient checks
@pytest.mark.parametrize("name,f,shapes,kw", [
    ("add_bcast", lambda a, b: (a + b).sum(), [(3, 4), (4,)], {}),
    ("add_bcast_col", lambda a, b: ((a + b) * (a + b)).sum(), [(3, 1), (1, 4)], {}),
    ("sub", lambda a, b: ((a - b) ** 2).mean(), [(2, 3), (2, 3)], {}),
    ("mul_bcast", lambda a, b: (a * b).sum(), [(2, 3, 4), (3, 1)], {}),
    ("div", lambda a, b: (a / b).sum(), [(3, 3), (3,)], {"positive": True}),
    ("pow", lambda a: (a ** 3).sum() + (a ** 0.5).sum(), [(4,)], {"positive": True}),
    ("neg", lambda a: (-a * a).sum(), [(3,)], {}),
    ("matmul", lambda a, b: (a @ b).sum(), [(3, 4), (4, 5)], {}),
    ("matmul_chain", lambda a, b, c: ((a @ b).tanh() @ c).sum(), [(2, 3), (3, 4), (4, 2)], {}),
    ("matmul_batched", lambda a, b: ((a @ b) ** 2).sum(), [(2, 3, 4), (4, 5)], {}),
    ("matmul_batched_both", lambda a, b: ((a @ b) ** 2).sum(), [(2, 3, 4), (2, 4, 2)], {}),
    ("exp", lambda a: a.exp().sum(), [(3, 2)], {}),
    ("log", lambda a: a.log().sum(), [(3, 2)], {"positive": True}),
    ("relu", lambda a: (a.relu() * a).sum(), [(10,)], {}),
    ("tanh", lambda a: a.tanh().sum(), [(5,)], {}),
    ("sigmoid", lambda a: (a.sigmoid() ** 2).sum(), [(5,)], {}),
    ("sum_axis0", lambda a: (a.sum(axis=0) ** 2).sum(), [(3, 4)], {}),
    ("sum_axis_neg_keep", lambda a: (a.sum(axis=-1, keepdims=True) * a).sum(), [(3, 4)], {}),
    ("sum_axes_tuple", lambda a: (a.sum(axis=(0, 2)) ** 2).sum(), [(2, 3, 4)], {}),
    ("mean", lambda a: (a.mean(axis=1) ** 2).sum() + a.mean(), [(3, 4)], {}),
    ("reshape", lambda a: (a.reshape(4, 3) @ a).sum(), [(3, 4)], {}),
    ("reshape_tuple_neg1", lambda a: (a.reshape((2, -1)) ** 2).sum(), [(3, 4)], {}),
    ("transpose", lambda a: (a.T @ a).sum(), [(3, 4)], {}),
    ("transpose_axes", lambda a: (a.transpose(2, 0, 1) ** 2 * np.arange(24).reshape(4, 2, 3)).sum(), [(2, 3, 4)], {}),
    ("getitem_slice", lambda a: (a[1:, ::2] ** 2).sum(), [(3, 4)], {}),
    ("getitem_fancy_repeat", lambda a: (a[np.array([0, 2, 0, 1]), np.array([1, 1, 1, 3])] * np.array([1.0, 2.0, 3.0, 4.0])).sum(), [(3, 4)], {}),
    ("log_softmax", lambda a: (a.log_softmax(axis=-1) * np.arange(12).reshape(3, 4)).sum(), [(3, 4)], {}),
    ("log_softmax_axis0", lambda a: (a.log_softmax(axis=0) ** 2).sum(), [(3, 4)], {}),
])
def test_op_gradients(name, f, shapes, kw):
    check(f, *shapes, **kw)


def test_softmax_cross_entropy_gradient_formula():
    """dL/dlogits for mean CE is (softmax - onehot) / N. Derive it -- it's on the quiz."""
    logits = rng.normal(size=(5, 4))
    y = np.array([0, 3, 1, 1, 2])
    x = Tensor(logits, requires_grad=True)
    loss = -x.log_softmax(axis=1)[np.arange(5), y].mean()
    loss.backward()
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    onehot = np.eye(4)[y]
    np.testing.assert_allclose(x.grad, (p - onehot) / 5, atol=1e-10)


def test_stability():
    x = Tensor(np.array([[1000.0, 0.0, -1000.0]]), requires_grad=True)
    ls = x.log_softmax(axis=-1)
    assert np.all(np.isfinite(ls.data))
    np.testing.assert_allclose(ls.data, [[0.0, -1000.0, -2000.0]], atol=1e-9)
    ls.sum().backward()
    assert np.all(np.isfinite(x.grad))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        s = Tensor(np.array([-1000.0, 0.0, 1000.0])).sigmoid()
    np.testing.assert_allclose(s.data, [0.0, 0.5, 1.0])


def test_mlp_end_to_end():
    """A 2-layer MLP with softmax cross-entropy -- the exact graph you'll train in weeks 2-3."""
    X = rng.normal(size=(6, 3))
    y = np.array([0, 1, 2, 1, 0, 2])

    def f(W1, b1, W2, b2):
        h = (X @ W1 + b1).tanh()
        logits = h @ W2 + b2
        return -logits.log_softmax(axis=-1)[np.arange(6), y].mean()
    check(f, (3, 5), (5,), (5, 3), (3,))
