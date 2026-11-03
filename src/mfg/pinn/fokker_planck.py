"""PINN for the wealth density m(t, x) of the population, under a given position pi*(t, x).

What the network learns
    The network N gives log m(t, x) = log m0(x) + t N(t, x). Then m = exp(...) is always
    positive, m(0, x) = m0(x) exactly (no loss needed for the start), and m stays tiny near the
    edges x_min and x_max, where m0 is (so the no-flux edges hold without a loss either).

What it is trained on: a loss with two parts, each a mean of squares
    equation  inside the domain, the Fokker-Planck residual m_t + (b m)_x - (s^2 m)_xx / 2, with
              drift b = r x + (mu - r) pi* and volatility s = sigma pi* (fp_residual). Extra points
              where the population lives.
    mass      the integral of m over x equals 1 at a few random times (trapezoid rule). The
              equation keeps the mass only if it is solved exactly; this part keeps it at 1.

The position enters through its derivatives
    The residual needs pi*_x and pi*_xx. Taking them through the value network would mean third
    and fourth derivatives of v, which are noisy near the kink. Instead, a second small network is
    fitted to pi* values by plain regression (fit_position_net) and differentiated: the fit
    smooths. Its error, weighted by the density, is at most about 0.002 (lam = 2, mbar = 1.1).

Reading the final mean Phi(mbar)
    The mean of the density drifts slowly with the leftover residual (+7.5e-4 at T, below). The mean
    equation dM/dt = r M + (mu - r) E_m[pi*] uses the density only as weights, which are far more
    accurate, and pi* itself, not its derivatives: DensityNetSolution.mean_by_equation. Use it for
    the equilibrium.

Accuracy (default settings, checked against the exact Gaussian and the FD solver, nx = 1600)
    lam = 0 (exact Merton pi*):  at T, mass 0.9997, mean -4e-4, variance -0.7%, density within
              1.2e-3 of the exact one (peak 1.10).
    lam = 2, mbar = 1.1, pi* from the value PINN through the position network:  density within
              0.09 for t <= 0.75 (peak 2.5 to 3), 0.3 at t = 0.9; near T the cliff at x = mbar
              (FD: from 4 to 0.6 within 0.01) is rounded, error up to 1.4 there. Phi: 1.06356
              directly, 1.062745 by the mean equation, against FD's 1.062810.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from mfg.params import Params
from mfg.pinn.nets import TanhNet, derivative, pick_device


def log_m0(x, p: Params):
    """Log of the initial density N(m0_mean, m0_std^2); works on tensors and arrays."""
    return -((x - p.m0_mean) ** 2) / (2 * p.m0_std**2) - np.log(p.m0_std * np.sqrt(2 * np.pi))


class DensityNet(nn.Module):
    """log m(t, x) = log m0(x) + t N(t, x), with N a TanhNet."""

    def __init__(self, p: Params, width: int = 64, depth: int = 4):
        super().__init__()
        self.p = p
        self.N = TanhNet(p, width, depth)

    def forward(self, t, x):
        return log_m0(x, self.p) + t * self.N(t, x)


def fp_residual(log_m_fn, position_fn, t, x, p: Params):
    """Fokker-Planck residual m_t + (b m)_x - (s^2 m)_xx / 2 at (t, x); 0 where m solves it.

    log_m_fn and position_fn are torch functions of (t, x); t and x are column tensors with
    requires_grad=True. Autograd differentiates through position_fn, which gives pi*_x and pi*_xx.
    """
    m = torch.exp(log_m_fn(t, x))
    pi = position_fn(t, x)
    flux_drift = (p.r * x + (p.mu - p.r) * pi) * m
    flux_diffusion = (p.sigma * pi) ** 2 * m
    return (
        derivative(m, t)
        + derivative(flux_drift, x)
        - 0.5 * derivative(derivative(flux_diffusion, x), x)
    )


# ---------------------------------------------------------------- the position network


@dataclass
class PositionNetSettings:
    """Fit of a smooth network to pi* values: plain least squares, no equation."""

    width: int = 64
    depth: int = 4
    n_points: int = 200_000  # (t, x) points, drawn once, with their pi* values
    share_domain: float = 0.2  # of the points: uniform on the whole domain...
    share_population: float = 0.4  # ...uniform where the population lives...
    population: tuple[float, float] = (0.0, 2.2)
    focus_dx: float = 0.3  # ...and the rest in [mbar - focus_dx, mbar + focus_dx], the dip
    batch: int = 4000
    adam_steps: int = 30_000
    lr_start: float = 1e-3
    lr_end: float = 1e-5
    lbfgs_steps: int = 500  # polish on a fixed subset of lbfgs_points (0 to skip)
    lbfgs_points: int = 50_000
    seed: int = 0
    device: str | None = None


def fit_position_net(
    position, p: Params, mbar, settings: PositionNetSettings | None = None, net=None
):
    """A frozen TanhNet fitted to position(t, x) (NumPy in and out, e.g. a solution's .position).

    net: a fitted network to start from (for example the one for a nearby mbar). It is copied
    first. Returns the network and the root-mean-square misfit on all the points.
    """
    s = settings or PositionNetSettings()
    device = s.device or pick_device()
    rng = np.random.default_rng(s.seed)
    torch.manual_seed(s.seed)
    n = s.n_points
    u = rng.random(n)
    x = np.where(
        u < s.share_domain,
        rng.uniform(p.x_min, p.x_max, n),
        np.where(
            u < s.share_domain + s.share_population,
            rng.uniform(*s.population, n),
            rng.uniform(mbar - s.focus_dx, mbar + s.focus_dx, n),
        ),
    )
    t = rng.uniform(0.0, p.T, n)
    y = np.concatenate(
        [position(t[i : i + 20_000], x[i : i + 20_000]) for i in range(0, n, 20_000)]
    )
    t, x, y = (
        torch.tensor(a, dtype=torch.float32, device=device).reshape(-1, 1) for a in (t, x, y)
    )

    net = TanhNet(p, s.width, s.depth) if net is None else copy.deepcopy(net)
    net = net.to(device)
    for w in net.parameters():
        w.requires_grad_(True)  # a fitted network comes frozen
    adam = torch.optim.Adam(net.parameters(), lr=s.lr_start)
    gamma = (s.lr_end / s.lr_start) ** (1 / max(s.adam_steps, 1))
    shrink = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=gamma)
    for _ in range(s.adam_steps):
        i = torch.randint(0, n, (s.batch,), device=device)
        loss = (net(t[i], x[i]) - y[i]).pow(2).mean()
        adam.zero_grad()
        loss.backward()
        adam.step()
        shrink.step()
    if s.lbfgs_steps > 0:
        j = torch.randperm(n, device=device)[: s.lbfgs_points]
        lbfgs = torch.optim.LBFGS(
            net.parameters(), max_iter=s.lbfgs_steps, history_size=50, line_search_fn="strong_wolfe"
        )

        def closure():
            lbfgs.zero_grad()
            loss = (net(t[j], x[j]) - y[j]).pow(2).mean()
            loss.backward()
            return loss

        lbfgs.step(closure)
    with torch.no_grad():
        rms = (net(t, x) - y).pow(2).mean().sqrt().item()
    for w in net.parameters():
        w.requires_grad_(False)
    return net, rms


# ---------------------------------------------------------------- the density network


@dataclass
class DensityNetSettings:
    """Size of the network, points per step and training choices."""

    width: int = 64
    depth: int = 4
    n_inside: int = 4000  # random points in the whole domain, new at every step
    n_population: int = 2000  # extra points where the population lives...
    population: tuple[float, float] = (0.0, 2.2)  # ...with x in this range
    n_mass_times: int = 16  # random times at which the mass is checked...
    n_mass_grid: int = 401  # ...by the trapezoid rule on this many points
    coef_equation: float = 1.0
    coef_mass: float = 1.0
    adam_steps: int = 10_000
    lr_start: float = 1e-3
    lr_end: float = 1e-4
    lbfgs_steps: int = 500
    seed: int = 0
    device: str | None = None
    print_every: int = 0


def sample_batch(p: Params, s: DensityNetSettings, device):
    """Fresh random points: inside (whole domain, then where the population lives) and the times
    at which the mass is checked."""

    def uniform(n, low, high):
        return low + (high - low) * torch.rand(n, 1, device=device)

    t = uniform(s.n_inside + s.n_population, 0.0, p.T)
    x = torch.cat([uniform(s.n_inside, p.x_min, p.x_max), uniform(s.n_population, *s.population)])
    return {"t": t, "x": x, "t_mass": uniform(s.n_mass_times, 0.0, p.T)}


def _mass_grid(p: Params, n, device):
    """Points and trapezoid weights on [x_min, x_max]."""
    x = torch.linspace(p.x_min, p.x_max, n, device=device).reshape(-1, 1)
    w = torch.full_like(x, (p.x_max - p.x_min) / (n - 1))
    w[0] = w[-1] = w[0] / 2
    return x, w


def density_net_loss(net, position_fn, batch, p: Params, s: DensityNetSettings):
    """The loss (the two parts times their coefficients) and the parts themselves."""
    t = batch["t"].clone().requires_grad_()
    x = batch["x"].clone().requires_grad_()
    equation = fp_residual(net, position_fn, t, x, p).pow(2).mean()

    x_grid, w = _mass_grid(p, s.n_mass_grid, t.device)
    t_mass = batch["t_mass"]
    k = len(t_mass)
    m = torch.exp(net(t_mass.repeat_interleave(len(x_grid), 0), x_grid.repeat(k, 1)))
    mass_at = (m.reshape(k, -1) * w.reshape(1, -1)).sum(dim=1)
    mass = (mass_at - 1.0).pow(2).mean()

    parts = {"equation": equation, "mass": mass}
    return s.coef_equation * equation + s.coef_mass * mass, parts


def train_density_net(p: Params, position_fn, settings: DensityNetSettings | None = None, net=None):
    """Train the density network under the frozen torch function position_fn(t, x).

    position_fn: e.g. the network from fit_position_net, or an exact formula in torch.
    net: a trained DensityNet to start from (it is copied first). Returns a DensityNetSolution.
    """
    s = settings or DensityNetSettings()
    device = s.device or pick_device()
    torch.manual_seed(s.seed)
    net = DensityNet(p, s.width, s.depth) if net is None else copy.deepcopy(net)
    net = net.to(device)
    history = {"step": [], "total": [], "equation": [], "mass": []}

    def record(step, total, parts):
        history["step"].append(step)
        history["total"].append(total.item())
        for name, value in parts.items():
            history[name].append(value.item())

    adam = torch.optim.Adam(net.parameters(), lr=s.lr_start)
    gamma = (s.lr_end / s.lr_start) ** (1 / max(s.adam_steps, 1))
    shrink = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=gamma)
    for step in range(s.adam_steps):
        total, parts = density_net_loss(net, position_fn, sample_batch(p, s, device), p, s)
        adam.zero_grad()
        total.backward()
        adam.step()
        shrink.step()
        if step % 100 == 0:
            record(step, total, parts)
        if s.print_every and step % s.print_every == 0:
            print(f"Adam step {step:6d}   loss {total.item():.2e}")

    if s.lbfgs_steps > 0:
        batch = sample_batch(p, s, device)
        lbfgs = torch.optim.LBFGS(
            net.parameters(), max_iter=s.lbfgs_steps, history_size=50, line_search_fn="strong_wolfe"
        )

        def closure():
            lbfgs.zero_grad()
            total, _ = density_net_loss(net, position_fn, batch, p, s)
            total.backward()
            return total

        lbfgs.step(closure)

    total, parts = density_net_loss(net, position_fn, sample_batch(p, s, device), p, s)
    record(s.adam_steps + s.lbfgs_steps, total, parts)
    if s.print_every:
        print(f"final loss {total.item():.2e}")
    return DensityNetSolution(net, position_fn, p, history)


@dataclass
class DensityNetSolution:
    """A trained density network, read with NumPy arrays that broadcast like t + x.

    density is the interface shared with the FD solver (CONTRIBUTING.md).
    """

    net: DensityNet
    position_fn: object  # the torch position function it was trained under
    p: Params
    history: dict

    def _evaluate(self, fn, t, x, grad=False):
        t, x = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(x, dtype=float))
        device = next(self.net.parameters()).device

        def column(a):
            return torch.tensor(
                a.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=grad
            )

        with torch.set_grad_enabled(grad):
            out = fn(column(t), column(x))
        return out.detach().cpu().numpy().astype(float).reshape(t.shape)

    def density(self, t, x):
        """m(t, x)."""
        return self._evaluate(lambda tt, xx: torch.exp(self.net(tt, xx)), t, x)

    def residual(self, t, x):
        """Fokker-Planck residual (see fp_residual): near 0 where the network solves it."""
        return self._evaluate(
            lambda tt, xx: fp_residual(self.net, self.position_fn, tt, xx, self.p), t, x, grad=True
        )

    def _grid(self, t, nx):
        x = np.linspace(self.p.x_min, self.p.x_max, nx)
        return x, self.density(np.asarray(t, dtype=float)[..., None], x)

    def mass(self, t, nx: int = 2001):
        """Integral of m(t, .) over the domain (trapezoid rule)."""
        x, m = self._grid(t, nx)
        return np.trapezoid(m, x, axis=-1)

    def mean(self, t, nx: int = 2001):
        """Mean wealth at time t, read directly from the density (divided by its mass)."""
        x, m = self._grid(t, nx)
        return np.trapezoid(m * x, x, axis=-1) / np.trapezoid(m, x, axis=-1)

    def mean_by_equation(self, position, nt: int = 401, nx: int = 2001):
        """Mean wealth at T from dM/dt = r M + (mu - r) E_m[pi*], M(0) = m0_mean.

        position(t, x) is pi* in NumPy (e.g. the value network's .position, not the fitted
        network): only the density's weights enter, which are more accurate than its mean.
        Trapezoid rule in time (exact for the linear r M term at this order).
        """
        p = self.p
        t = np.linspace(0.0, p.T, nt)
        x, m = self._grid(t, nx)
        weights = m / np.trapezoid(m, x, axis=-1)[:, None]
        # One time at a time: a value network's position needs autograd, too big for all at once.
        expected_pi = np.array(
            [
                np.trapezoid(w * position(np.full_like(x, ti), x), x)
                for ti, w in zip(t, weights, strict=True)
            ]
        )
        mean = p.m0_mean
        for i in range(nt - 1):
            h = t[i + 1] - t[i]
            push = 0.5 * h * (p.mu - p.r) * (expected_pi[i] + expected_pi[i + 1])
            mean = (mean * (1 + 0.5 * h * p.r) + push) / (1 - 0.5 * h * p.r)
        return mean
