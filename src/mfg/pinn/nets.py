"""Pieces shared by the PINNs: tanh networks of (t, x) and of (t, x, rivals), and exact
derivatives of them."""

import torch
from torch import nn

from mfg.params import Params


def tanh_stack(n_in, width, depth, n_out=1):
    """Linear layers with tanh between them, Xavier start and zero biases.

    tanh is smooth and its second derivative is not zero, which the PINNs need: their equations
    contain second derivatives of the network, and the optimal position depends on one. With
    ReLU the second derivative would be zero everywhere. Xavier keeps the signal of a similar
    size through every tanh layer.
    """
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(n_in, width), nn.Tanh()]
        n_in = width
    layers.append(nn.Linear(n_in, n_out))
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers)


class TanhNet(nn.Module):
    """Fully connected network that maps (t, x) to one number, with tanh activations."""

    def __init__(self, p: Params, width: int = 64, depth: int = 4):
        super().__init__()
        # The network sees t and x rescaled from [0, T] x [x_min, x_max] to [-1, 1] x [-1, 1],
        # where tanh is not saturated.
        self.t_scale = 2.0 / p.T
        self.x_mid = 0.5 * (p.x_min + p.x_max)
        self.x_scale = 2.0 / (p.x_max - p.x_min)
        self.layers = tanh_stack(2, width, depth)

    def forward(self, t, x):
        """t and x are column tensors of shape (n, 1); the result has shape (n, 1)."""
        scaled = torch.cat([self.t_scale * t - 1.0, self.x_scale * (x - self.x_mid)], dim=1)
        return self.layers(scaled)


class SymmetricNet(nn.Module):
    """Network of (t, x, y) for the finite-N game: x is my wealth and y the wealths of my N - 1
    rivals, whose order must not matter.

    Each rival's wealth goes through one small summary network (the same weights for every
    rival) and the summaries are averaged: the average does not change when the rivals are
    reordered, and it keeps the same size for any N, so the same weights serve any number of
    rivals (N = 20 can start from the network of N = 10). The final network then sees t, x, the
    rivals' mean ybar (given directly, because the kink of the final utility sits at x = ybar)
    and the averaged summary. Any symmetric function can be written this way with a summary of
    about N + 1 numbers (DeepSets; Germain, Lauriere, Pham and Warin 2022): 32 covers N <= 20.
    Inputs are rescaled as in TanhNet; tanh everywhere, since the equation needs second
    derivatives in every wealth.
    """

    def __init__(
        self,
        p: Params,
        features: int = 32,
        summary_width: int = 32,
        width: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        # Kept so that a saved network can be rebuilt (nplayer_io.py).
        self.kwargs = dict(features=features, summary_width=summary_width, width=width, depth=depth)
        self.t_scale = 2.0 / p.T
        self.x_mid = 0.5 * (p.x_min + p.x_max)
        self.x_scale = 2.0 / (p.x_max - p.x_min)
        self.summary = tanh_stack(1, summary_width, 2, features)
        self.final = tanh_stack(3 + features, width, depth)

    def forward(self, t, x, y):
        """t and x have shape (n, 1), y has shape (n, N - 1); the result has shape (n, 1)."""
        n, n_rivals = y.shape
        y_scaled = self.x_scale * (y - self.x_mid)
        summary = self.summary(y_scaled.reshape(-1, 1)).reshape(n, n_rivals, -1).mean(dim=1)
        inputs = torch.cat(
            [
                self.t_scale * t - 1.0,
                self.x_scale * (x - self.x_mid),
                y_scaled.mean(dim=1, keepdim=True),
                summary,
            ],
            dim=1,
        )
        return self.final(inputs)


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
