"""Closed forms used to validate the solvers.

With lam = 0 every investor plays Merton, so V is explicit and the wealth
distribution stays Gaussian with explicit mean and variance.

For lam > 0 there is no closed form for V, but the expected utility of simply
holding the Merton position is explicit; it gives the HJB boundary value at x_min.
"""

import numpy as np
from scipy.stats import norm

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


def merton_strategy_value(t, x, mbar, p: Params):
    """E[U(X_T, mbar) | X_t = x] for an investor who holds the Merton position until T.

    X_T is then Gaussian with mean M = x e^{r(T-t)} + kappa^2 (T-t) / gamma and variance
    S^2 = kappa^2 (T-t) / gamma^2, so both parts of the shortfall utility have closed forms.
    The strategy is not optimal for lam > 0, so this is a lower bound on V; both the FD and the
    PINN solvers use it as the boundary value at x_min. At lam = 0 it equals merton_value.
    """
    tau = p.T - t
    mean = x * np.exp(p.r * tau) + p.kappa**2 / p.gamma * tau
    std = p.kappa / p.gamma * np.sqrt(tau)
    d = mbar - mean
    cara = -np.exp(-p.gamma * mean + 0.5 * (p.gamma * std) ** 2)
    # E[((mbar - X_T)^+)^2]; at t = T (std = 0) it is just the squared shortfall.
    with np.errstate(divide="ignore", invalid="ignore"):
        z = d / std
        shortfall_sq = (d**2 + std**2) * norm.cdf(z) + d * std * norm.pdf(z)
    shortfall_sq = np.where(std > 0, shortfall_sq, np.maximum(d, 0.0) ** 2)
    return cara - 0.5 * p.lam * shortfall_sq


# ---------------------------------------------------------------- the finite-N game


def lambda_n(lam, N):
    """Penalty strength seen against the mean of my N - 1 rivals: lam (1 - 1/N)^2.

    The group mean includes me, so (xbar - x) = (1 - 1/N)(ybar - x) with ybar the rivals' mean;
    the penalty (lam/2)((xbar - x)^+)^2 is then (lam_N/2)((ybar - x)^+)^2.
    """
    return lam * (1.0 - 1.0 / N) ** 2


def c2_k(N, eta):
    """Constant k_N of the C2 closed form: k_N = -1/2 + (N - 1)[a + a^2/2], a = eta / (N - eta).

    It comes out of the HJB when the linear v below is plugged in: the x and y terms cancel and
    the constant left fixes k_N. With eta = 0 it is -1/2 (Merton).
    """
    a = eta / (N - eta)
    return -0.5 + (N - 1) * (a + 0.5 * a**2)


def c2_position(t, N, eta, p: Params):
    """Exact position in the N-player game with the original utility -exp(-gamma (x - eta xbar)).

    Everyone plays Merton with risk aversion gamma (1 - eta/N): comparing myself with a mean that
    I am 1/N of is partly comparing myself with myself. (Lacker and Zariphopoulou 2019, without
    common noise, plus the e^{-r(T-t)} of our model with r.)
    """
    return (p.mu - p.r) * np.exp(-p.r * (p.T - t)) / (p.gamma * (1.0 - eta / N) * p.sigma**2)


def c2_value(t, x, y, N, eta, p: Params):
    """Exact V(t, x, y) of the game above, like merton_value: V = -exp(v) with

    v = -gamma e^{r(T-t)} [(1 - eta/N) x - (eta/N) sum_j y_j] + kappa^2 k_N (T - t).

    x broadcasts with t; y has the N - 1 rivals' wealths on its last axis.
    """
    t, x = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(x, dtype=float))
    tau = p.T - t
    linear = (1.0 - eta / N) * x - eta / N * np.asarray(y, dtype=float).sum(axis=-1)
    return -np.exp(-p.gamma * np.exp(p.r * tau) * linear + p.kappa**2 * c2_k(N, eta) * tau)
