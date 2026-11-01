"""Pieces shared by the PINNs: a small tanh network of (t, x) and exact derivatives of it."""

import torch
from torch import nn

from mfg.params import Params


class TanhNet(nn.Module):
    """Fully connected network that maps (t, x) to one number, with tanh activations.

    tanh is smooth and its second derivative is not zero, which the PINNs need: their equations
    contain second derivatives of the network, and the optimal position depends on one. With
    ReLU the second derivative would be zero everywhere.
    """

    def __init__(self, p: Params, width: int = 64, depth: int = 4):
        super().__init__()
        # The network sees t and x rescaled from [0, T] x [x_min, x_max] to [-1, 1] x [-1, 1],
        # where tanh is not saturated.
        self.t_scale = 2.0 / p.T
        self.x_mid = 0.5 * (p.x_min + p.x_max)
        self.x_scale = 2.0 / (p.x_max - p.x_min)
        layers, n_in = [], 2
        for _ in range(depth):
            layers += [nn.Linear(n_in, width), nn.Tanh()]
            n_in = width
        layers.append(nn.Linear(n_in, 1))
        self.layers = nn.Sequential(*layers)
        # Xavier start: keeps the signal of a similar size through every tanh layer.
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, t, x):
        """t and x are column tensors of shape (n, 1); the result has shape (n, 1)."""
        scaled = torch.cat([self.t_scale * t - 1.0, self.x_scale * (x - self.x_mid)], dim=1)
        return self.layers(scaled)


def derivative(y, z):
    """dy/dz at every point, kept differentiable so that it can be differentiated again.

    y and z are column tensors of the same length; z must have requires_grad=True. If y does not
    depend on z (for example the slope of a function that is straight in x), the result is 0.
    """
    if not y.requires_grad:
        return torch.zeros_like(z)
    grad = torch.autograd.grad(y, z, torch.ones_like(y), create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(z) if grad is None else grad


def pick_device():
    """The GPU if PyTorch can see one, otherwise the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"
