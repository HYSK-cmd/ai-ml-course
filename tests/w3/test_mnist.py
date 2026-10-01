"""Week 3 slow tests: train real networks on MNIST with forge (numpy, CPU).
Run with:  pytest tests/w3 -m slow -s     (-s shows the progress prints)"""
import time

import numpy as np
import pytest

from forge.autograd import Tensor, no_grad
from forge.nn import (BatchNorm1d, Conv2d, DataLoader, Dropout, Flatten, Linear, MaxPool2d, ReLU, Sequential,
                      cross_entropy)
from forge.optim import Adam
from tests.helpers import need_data

pytestmark = pytest.mark.slow


def mnist(flat=True):
    (path,) = need_data("mnist.npz")
    d = np.load(path)
    f = (lambda X: X.reshape(-1, 784)) if flat else (lambda X: X.reshape(-1, 1, 28, 28))
    return (f(d["X_train"].astype(np.float32) / 255), d["y_train"].astype(int),
            f(d["X_test"].astype(np.float32) / 255), d["y_test"].astype(int))


def fit(model, Xtr, ytr, epochs, batch_size, lr, n_train=None):
    opt = Adam(model.parameters(), lr=lr)
    if n_train:
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    for ep in range(epochs):
        model.train()
        t = time.perf_counter()
        for xb, yb in DataLoader(Xtr, ytr, batch_size=batch_size, shuffle=True, seed=ep):
            opt.zero_grad()
            loss = cross_entropy(model(Tensor(xb)), yb)
            loss.backward()
            opt.step()
        print(f"  epoch {ep + 1}: last loss {float(loss.data):.4f} ({time.perf_counter() - t:.1f}s)")


def accuracy(model, X, y, batch_size=1000):
    model.eval()
    correct = 0
    with no_grad():
        for xb, yb in DataLoader(X, y, batch_size=batch_size):
            correct += (model(Tensor(xb)).data.argmax(1) == yb).sum()
    return correct / len(y)


def test_mlp_mnist():
    Xtr, ytr, Xte, yte = mnist()
    rng = np.random.default_rng(0)
    model = Sequential(Linear(784, 256, rng=rng), ReLU(), Linear(256, 10, rng=rng))
    fit(model, Xtr, ytr, epochs=5, batch_size=128, lr=1e-3)
    acc = accuracy(model, Xte, yte)
    print(f"MLP test accuracy {acc:.4f}")
    assert acc >= 0.972  # PyTorch with this exact recipe: ~0.975


def test_bn_dropout_mlp_mnist():
    Xtr, ytr, Xte, yte = mnist()
    rng = np.random.default_rng(0)
    model = Sequential(Linear(784, 256, rng=rng), BatchNorm1d(256), ReLU(), Dropout(0.2, rng=rng),
                       Linear(256, 10, rng=rng))
    fit(model, Xtr, ytr, epochs=3, batch_size=128, lr=1e-3)
    acc = accuracy(model, Xte, yte)
    print(f"BN+Dropout MLP test accuracy {acc:.4f}")
    assert acc >= 0.972, "if train looks fine but test is bad, check eval() mode in BatchNorm/Dropout"


def test_cnn_mnist():
    Xtr, ytr, Xte, yte = mnist(flat=False)
    rng = np.random.default_rng(0)
    model = Sequential(
        Conv2d(1, 8, 3, padding=1, rng=rng), ReLU(), MaxPool2d(2),
        Conv2d(8, 16, 3, padding=1, rng=rng), ReLU(), MaxPool2d(2),
        Flatten(), Linear(16 * 7 * 7, 10, rng=rng))
    t = time.perf_counter()
    fit(model, Xtr, ytr, epochs=3, batch_size=64, lr=2e-3)
    elapsed = time.perf_counter() - t
    acc = accuracy(model, Xte, yte)
    print(f"CNN test accuracy {acc:.4f}, 3 epochs in {elapsed:.0f}s")
    assert acc >= 0.978  # PyTorch with this exact recipe: ~0.985
    assert elapsed < 1800, "3 CNN epochs must finish within 30 minutes"
