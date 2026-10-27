"""Closed-form solution of the game without interaction (lam = 0), used to validate solvers.

With lam = 0 every investor plays Merton, so V is explicit and the wealth
distribution stays Gaussian with explicit mean and variance.
"""

import numpy as np

from mfg.params import Params


def merton_pi(t, p: Params):
    """Optimal dollar position pi*(t) = (mu - r) e^{-r(T-t)} / (gamma sigma^2)."""
    return (p.mu - p.r) * np.exp(-p.r * (p.T - t)) / (p.gamma * p.sigma**2)


def merton_value(t, x, p: Params):
    """V(t, x) = -exp(-gamma x e^{r(T-t)} - kappa^2 (T-t) / 2)."""
    tau = p.T - t
    return -np.exp(-p.gamma * x * np.exp(p.r * tau) - 0.5 * p.kappa**2 * tau)


def population_mean(t, p: Params):
    """Mean of the wealth distribution m_t when everyone plays Merton."""
    return p.m0_mean * np.exp(p.r * t) + p.kappa**2 / p.gamma * t * np.exp(p.r * (t - p.T))


def population_var(t, p: Params):
    """Variance of m_t when everyone plays Merton."""
    return p.m0_std**2 * np.exp(2 * p.r * t) + (p.kappa / p.gamma) ** 2 * t * np.exp(
        2 * p.r * (t - p.T)
    )
