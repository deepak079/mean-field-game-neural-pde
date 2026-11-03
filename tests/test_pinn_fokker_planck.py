"""Tests for the density PINN (src/mfg/pinn/fokker_planck.py): the residual against an exact
solution and against finite differences, the position fit, the mean equation, and a short training
run against the lam = 0 Gaussian."""

import numpy as np
import torch

from mfg.benchmarks import merton_pi, population_mean, population_var
from mfg.params import Params
from mfg.pinn.fokker_planck import (
    DensityNetSettings,
    PositionNetSettings,
    fit_position_net,
    fp_residual,
    train_density_net,
)
from mfg.pinn.nets import derivative

P0 = Params().with_(lam=0.0)


def columns(t, x):
    t, x = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(x, dtype=float))

    def as_column(a):
        return torch.tensor(a.reshape(-1, 1), dtype=torch.float64).requires_grad_()

    return as_column(t), as_column(x)


def merton_position(t, x):
    """pi* = 1.5 e^{-r (T - t)} in torch, the same at every x."""
    return (P0.mu - P0.r) / (P0.gamma * P0.sigma**2) * torch.exp(-P0.r * (P0.T - t)) + 0 * x


def gaussian_log_m(t, x):
    """Exact log m at lam = 0: Gaussian with the mean and variance of benchmarks.py, in torch."""
    tau = P0.T - t
    mean = P0.m0_mean * torch.exp(P0.r * t) + P0.kappa**2 / P0.gamma * t * torch.exp(-P0.r * tau)
    var = P0.m0_std**2 * torch.exp(2 * P0.r * t) + (P0.kappa / P0.gamma) ** 2 * t * torch.exp(
        -2 * P0.r * tau
    )
    return -((x - mean) ** 2) / (2 * var) - 0.5 * torch.log(2 * np.pi * var)


T_GRID, X_GRID = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.2, 12))


def test_gaussian_has_zero_residual_at_lam_0():
    t, x = columns(T_GRID, X_GRID)
    residual = fp_residual(gaussian_log_m, merton_position, t, x, P0)
    assert residual.abs().max().item() < 1e-10


def test_residual_matches_finite_differences_when_pi_depends_on_x():
    # Any smooth m and pi will do: the residual of autograd against central differences of
    # m_t + (b m)_x - (s^2 m)_xx / 2 written out with NumPy.
    def log_m(t, x):
        return -((x - 1 - 0.1 * t) ** 2) / (2 * (0.04 + 0.1 * t))

    def position(t, x):
        return 1 + 0.5 * torch.sin(2 * x) + 0.3 * t

    p = Params()
    t, x = columns(T_GRID, X_GRID)
    by_autograd = fp_residual(log_m, position, t, x, p).detach().numpy().ravel()

    def m_np(t, x):
        return np.exp(-((x - 1 - 0.1 * t) ** 2) / (2 * (0.04 + 0.1 * t)))

    def pi_np(t, x):
        return 1 + 0.5 * np.sin(2 * x) + 0.3 * t

    def drift_flux(t, x):
        return (p.r * x + (p.mu - p.r) * pi_np(t, x)) * m_np(t, x)

    def diffusion_flux(t, x):
        return (p.sigma * pi_np(t, x)) ** 2 * m_np(t, x)

    tt, xx, h = T_GRID.ravel(), X_GRID.ravel(), 1e-4
    by_hand = (
        (m_np(tt + h, xx) - m_np(tt - h, xx)) / (2 * h)
        + (drift_flux(tt, xx + h) - drift_flux(tt, xx - h)) / (2 * h)
        - 0.5
        * (diffusion_flux(tt, xx + h) - 2 * diffusion_flux(tt, xx) + diffusion_flux(tt, xx - h))
        / h**2
    )
    np.testing.assert_allclose(by_autograd, by_hand, atol=1e-5)


SMALL_FIT = dict(width=16, depth=2, n_points=5000, adam_steps=500, lbfgs_steps=50, device="cpu")


def test_position_net_fits_a_known_position_and_is_frozen():
    net, rms = fit_position_net(
        lambda t, x: merton_pi(t, P0) + 0 * x, P0, 1.1, PositionNetSettings(**SMALL_FIT)
    )
    assert rms < 0.003  # observed 1e-3
    assert not any(w.requires_grad for w in net.parameters())
    # Its x-derivative, which the residual uses, is close to the exact 0.
    t, x = columns(T_GRID, X_GRID)
    slope = derivative(net(t.float(), x.float()), x).detach().numpy()
    assert np.abs(slope).max() < 0.01  # observed 0.003


SMALL = dict(
    width=32,
    depth=3,
    n_inside=1000,
    n_population=500,
    n_mass_times=8,
    n_mass_grid=201,
    device="cpu",
)


def test_untrained_density_starts_at_m0_and_mean_equation_is_exact_at_lam_0():
    # At lam = 0, pi* does not depend on x, so E_m[pi*] = pi* for any m with mass 1: the mean
    # equation gives the exact mean even from an untrained network.
    sol = train_density_net(
        P0, merton_position, DensityNetSettings(adam_steps=0, lbfgs_steps=0, **SMALL)
    )
    x = np.linspace(-1.0, 3.0, 9)
    m0 = np.exp(-((x - 1) ** 2) / 0.08) / (0.2 * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(sol.density(0.0, x), m0, rtol=1e-5, atol=1e-12)
    mean_T = sol.mean_by_equation(lambda t, x: merton_pi(t, P0) + 0 * x)
    assert abs(mean_T - population_mean(P0.T, P0)) < 1e-6


def test_density_net_learns_the_gaussian_without_penalty():
    # A small network on the CPU (about 25 s). Observed at t = 1: mass 6e-5, mean 1.9e-3,
    # density 2.1e-3 (peak 1.1); residual 0.011 at t = 0.5. Margins about three times.
    sol = train_density_net(
        P0, merton_position, DensityNetSettings(adam_steps=1500, lbfgs_steps=200, **SMALL)
    )
    for t in (0.5, 1.0):
        assert abs(sol.mass(t) - 1) < 2e-4
        assert abs(sol.mean(t) - population_mean(t, P0)) < 0.006
        x = np.linspace(0.0, 2.2, 23)
        mean, var = population_mean(t, P0), population_var(t, P0)
        exact = np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
        np.testing.assert_allclose(sol.density(t, x), exact, atol=0.007)
    assert np.abs(sol.residual(0.5, np.linspace(0.0, 2.2, 12))).max() < 0.04


def test_warm_start_leaves_the_earlier_solution_untouched():
    first = train_density_net(
        P0, merton_position, DensityNetSettings(adam_steps=0, lbfgs_steps=0, **SMALL)
    )
    before = first.density(0.5, np.linspace(-1.0, 3.0, 5))
    train_density_net(
        P0,
        merton_position,
        DensityNetSettings(adam_steps=20, lbfgs_steps=0, **SMALL),
        net=first.net,
    )
    np.testing.assert_array_equal(first.density(0.5, np.linspace(-1.0, 3.0, 5)), before)
