"""Week 3 autograder: Module system, state_dict, DataLoader."""
import numpy as np
import pytest

from forge.autograd import Tensor
from forge.nn import (BatchNorm1d, DataLoader, Dropout, Linear, Module, Parameter, ReLU, Sequential)


class Block(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(4, 3)
        self.bn = BatchNorm1d(3)
        self.scale = Parameter(np.ones(1))

    def forward(self, x):
        return self.bn(self.fc(x)).relu() * self.scale


class Net(Module):
    def __init__(self):
        super().__init__()
        self.encoder = Block()
        self.head = Sequential(Linear(3, 5), ReLU(), Linear(5, 2))
        self.not_a_param = np.zeros(3)  # plain arrays are NOT parameters

    def forward(self, x):
        return self.head(self.encoder(x))


def test_parameter_is_trainable_leaf():
    p = Parameter(np.zeros(3))
    assert isinstance(p, Tensor) and p.requires_grad


def test_registration_and_names():
    net = Net()
    names = [n for n, _ in net.named_parameters()]
    assert names == ["encoder.scale", "encoder.fc.weight", "encoder.fc.bias", "encoder.bn.weight",
                     "encoder.bn.bias", "head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias"]
    assert len(net.parameters()) == 9
    assert [n for n, _ in net.named_buffers()] == ["encoder.bn.running_mean", "encoder.bn.running_var"]
    assert len(list(net.children())) == 2
    assert len(net.head) == 3 and isinstance(net.head[1], ReLU)


def test_shared_parameter_counted_once():
    class Tied(Module):
        def __init__(self):
            super().__init__()
            self.a = Linear(3, 3)
            self.b = Linear(3, 3)
            self.b.weight = self.a.weight

        def forward(self, x):
            return self.b(self.a(x))
    t = Tied()
    assert len(t.parameters()) == 3  # a.weight (shared), a.bias, b.bias
    assert t.b.weight is t.a.weight


def test_reassignment_replaces():
    m = Linear(2, 2)
    m.bias = Parameter(np.ones(2))
    assert len(m.parameters()) == 2
    np.testing.assert_array_equal(dict(m.named_parameters())["bias"].data, np.ones(2))


def test_train_eval_recursive():
    net = Net()
    assert net.training and net.encoder.bn.training
    assert net.eval() is net
    assert not net.training and not net.encoder.bn.training and not net.head[0].training
    net.train()
    assert net.head[2].training


def test_zero_grad_and_backward_reach_all_params():
    net = Net()
    x = Tensor(np.random.default_rng(0).normal(size=(8, 4)))
    net(x).sum().backward()
    assert all(p.grad is not None for p in net.parameters())
    net.zero_grad()
    assert all(p.grad is None for p in net.parameters())


def test_state_dict_roundtrip_and_errors():
    a, b = Net(), Net()
    a(Tensor(np.random.default_rng(1).normal(size=(16, 4))))  # changes BN running stats
    sd = a.state_dict()
    assert set(sd) == {n for n, _ in a.named_parameters()} | {n for n, _ in a.named_buffers()}
    sd["head.0.bias"][0] = 123.0
    assert a.head[0].bias.data[0] != 123.0, "state_dict must return copies"
    b.load_state_dict(a.state_dict())
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        np.testing.assert_array_equal(p1.data, p2.data)
    np.testing.assert_allclose(a.encoder.bn.running_mean, b.encoder.bn.running_mean, rtol=1e-6)
    bad = a.state_dict()
    bad.pop("head.2.bias")
    with pytest.raises(KeyError):
        b.load_state_dict(bad)
    bad = a.state_dict()
    bad["extra"] = np.zeros(1)
    with pytest.raises(KeyError):
        b.load_state_dict(bad)
    bad = a.state_dict()
    bad["head.2.bias"] = np.zeros(7)
    with pytest.raises(ValueError):
        b.load_state_dict(bad)


def test_load_state_dict_is_in_place():
    m = Linear(3, 2)
    w = m.weight
    m.load_state_dict({"weight": np.ones((2, 3)), "bias": np.zeros(2)})
    assert m.weight is w, "optimizers hold references to the Parameter objects -- copy values in place"
    np.testing.assert_array_equal(w.data, np.ones((2, 3)))


def test_to_dtype():
    net = Net().to(np.float64)
    assert all(p.data.dtype == np.float64 for p in net.parameters())
    assert net.encoder.bn.running_var.dtype == np.float64
    assert Linear(2, 2).weight.data.dtype == np.float32, "default parameter dtype is float32"


def test_dataloader():
    X = np.arange(10)[:, None].astype(float)
    y = np.arange(10)
    dl = DataLoader(X, y, batch_size=4)
    batches = list(dl)
    assert len(dl) == 3 and [len(b[1]) for b in batches] == [4, 4, 2]
    np.testing.assert_array_equal(np.concatenate([b[1] for b in batches]), y)
    assert len(DataLoader(X, y, batch_size=4, drop_last=True)) == 2
    assert [len(b) for _, b in DataLoader(X, y, batch_size=4, drop_last=True)] == [4, 4]
    dl = DataLoader(X, y, batch_size=3, shuffle=True, seed=0)
    e1 = np.concatenate([b for _, b in dl])
    e2 = np.concatenate([b for _, b in dl])
    assert sorted(e1) == list(range(10)) and not np.array_equal(e1, e2), "reshuffle every epoch"
    for xb, yb in dl:
        np.testing.assert_array_equal(xb[:, 0], yb)
    e1b = np.concatenate([b for _, b in DataLoader(X, y, batch_size=3, shuffle=True, seed=0)])
    np.testing.assert_array_equal(e1, e1b)


def test_dropout_modes():
    d = Dropout(0.25, rng=np.random.default_rng(0))
    x = Tensor(np.ones((200, 200)))
    out = d(x).data
    frac_zero = (out == 0).mean()
    assert 0.23 < frac_zero < 0.27
    np.testing.assert_allclose(out[out != 0], 1 / 0.75)
    assert abs(out.mean() - 1) < 0.02
    d.eval()
    np.testing.assert_array_equal(d(x).data, x.data)
