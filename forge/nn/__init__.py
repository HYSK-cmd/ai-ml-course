"""Week 3 -- a tiny PyTorch-like neural-net library on top of YOUR forge.autograd."""
from forge.nn.conv import Conv2d, MaxPool2d
from forge.nn.data import DataLoader
from forge.nn.layers import (
    BatchNorm1d,
    Dropout,
    Flatten,
    LayerNorm,
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
)
from forge.nn.losses import cross_entropy, mse_loss
from forge.nn.module import Module, Parameter, Sequential

__all__ = [
    "BatchNorm1d", "Conv2d", "DataLoader", "Dropout", "Flatten", "LayerNorm", "Linear", "MaxPool2d", "Module",
    "Parameter", "ReLU", "Sequential", "Sigmoid", "Tanh", "cross_entropy", "mse_loss",
]
