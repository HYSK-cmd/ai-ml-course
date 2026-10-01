"""Week 1 autograder: forge.optim + forge.gradcheck."""
import numpy as np
import pytest

from forge.autograd import Tensor
from forge.gradcheck import gradcheck, numerical_grad
from forge.optim import SGD, Adam, AdamW

# Expected parameters after 3 steps, generated with PyTorch's update rules.
EXPECTED = {
    ("SGD", (("lr", 0.1),)): [0.46783463619892995, -0.5279470562995199, 1.6876824275517297],
    ("SGD", (("lr", 0.1), ("momentum", 0.9))): [-0.032529328341069974, 0.8379734915404802, 0.44011833289172975],
    ("SGD", (("lr", 0.1), ("momentum", 0.9), ("weight_decay", 0.01))): [-0.03608023505865887, 0.8431774890184907, 0.4286891584011179],
    ("Adam", (("lr", 0.1),)): [0.7018462483868272, -1.700501927117109, 2.700595671390876],
    ("AdamW", (("lr", 0.1), ("weight_decay", 0.1))): [0.6754002798409069, -1.6442148480513263, 2.614707341381231],
}
CLASSES = {"SGD": SGD, "Adam": Adam, "AdamW": AdamW}


@pytest.mark.parametrize("key", list(EXPECTED), ids=lambda k: f"{k[0]}{dict(k[1])}")
def test_three_steps_match_pytorch(key):
    name, kw = key
    x = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)
    opt = CLASSES[name]((p for p in [x]), **dict(kw))  # generators must work
    for _ in range(3):
        opt.zero_grad()
        loss = (x * x * np.array([1.0, 2.0, 0.5])).sum() + (x ** 3).sum() * 0.1
        loss.backward()
        opt.step()
    np.testing.assert_allclose(x.data, EXPECTED[key], rtol=1e-9)


def test_zero_grad_and_skip_none():
    a = Tensor(np.ones(2), requires_grad=True)
    b = Tensor(np.ones(2), requires_grad=True)
    opt = Adam([a, b], lr=0.1)
    (a * 2).sum().backward()
    opt.step()  # b.grad is None -> b must not change (and must not crash)
    np.testing.assert_allclose(b.data, np.ones(2))
    assert not np.allclose(a.data, np.ones(2))
    opt.zero_grad()
    assert a.grad is None and b.grad is None


def test_updates_in_place():
    p = Tensor(np.ones(3), requires_grad=True)
    ref = p.data
    opt = SGD([p], lr=0.5)
    (p * p).sum().backward()
    opt.step()
    assert p.data is ref, "update p.data in place (p.data -= ...), don't rebind it"


def test_adam_minimizes_rosenbrock():
    xy = Tensor(np.array([-1.5, 2.0]), requires_grad=True)
    opt = Adam([xy], lr=0.02)
    for _ in range(5000):
        opt.zero_grad()
        x, y = xy[0], xy[1]
        loss = (1 - x) ** 2 + 100 * (y - x * x) ** 2
        loss.backward()
        opt.step()
    np.testing.assert_allclose(xy.data, [1.0, 1.0], atol=2e-2)


def test_sgd_linear_regression():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    w_true = np.array([2.0, -1.0, 0.5])
    y = X @ w_true + 0.3
    w = Tensor(np.zeros(3), requires_grad=True)
    b = Tensor(0.0, requires_grad=True)
    opt = SGD([w, b], lr=0.1, momentum=0.5)
    for _ in range(300):
        opt.zero_grad()
        loss = ((X @ w.reshape(3, 1)).reshape(200) + b - y) ** 2
        loss.mean().backward()
        opt.step()
    np.testing.assert_allclose(w.data, w_true, atol=1e-3)
    assert abs(float(b.data) - 0.3) < 1e-3


# ---------------------------------------------------------------- gradcheck
def test_numerical_grad_values_and_no_side_effects():
    x = np.array([1.0, 2.0, -3.0])
    x0 = x.copy()
    (g,) = numerical_grad(lambda t: (t * t * t).sum(), [x])
    np.testing.assert_allclose(g, 3 * x0 ** 2, rtol=1e-6)
    np.testing.assert_array_equal(x, x0)


def test_gradcheck_accepts_correct_rejects_wrong():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(4, 2))
    assert gradcheck(lambda a, b: ((a @ b).tanh() ** 2).sum(), [A, B])
    # `.data` leaks out of the graph: autograd sees x * const, but the true function is x**2
    assert not gradcheck(lambda a: (a * Tensor(a.data)).sum(), [A])
