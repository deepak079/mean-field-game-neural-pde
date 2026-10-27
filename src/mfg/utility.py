"""Terminal utility U(x, mbar) = -exp(-gamma x) - (lam/2) ((mbar - x)^+)^2 and its derivatives.

Works on NumPy arrays; the PINN code can reuse the same formulas with torch tensors.
"""

import numpy as np

from mfg.params import Params


def utility(x, mbar, p: Params):
    gap = np.maximum(mbar - x, 0.0)
    return -np.exp(-p.gamma * x) - 0.5 * p.lam * gap**2


def utility_x(x, mbar, p: Params):
    gap = np.maximum(mbar - x, 0.0)
    return p.gamma * np.exp(-p.gamma * x) + p.lam * gap


def utility_xx(x, mbar, p: Params):
    below = np.where(x < mbar, 1.0, 0.0)
    return -(p.gamma**2) * np.exp(-p.gamma * x) - p.lam * below


def pi_terminal(x, mbar, p: Params):
    """Exact optimal position at t = T: (mu - r)/sigma^2 * (-U_x / U_xx)."""
    return (p.mu - p.r) / p.sigma**2 * utility_x(x, mbar, p) / -utility_xx(x, mbar, p)
