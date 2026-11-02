"""Finite-difference solver for the HJB with a fixed terminal mean mbar.

Implicit Euler backward from T, with policy iteration inside each time step: fix pi, solve
the linear tridiagonal system, update pi, repeat. The steps follow Appendix A of
DS's notes on the HJB and the Fokker-Planck equation (kept outside the repo).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_banded

from mfg.benchmarks import merton_strategy_value, merton_value
from mfg.fd.generator import jump_rates
from mfg.params import Params
from mfg.utility import pi_terminal, utility


@dataclass
class HJBSolution:
    """V and pi* on the grid, plus interpolation for evaluation anywhere in the domain."""

    t: np.ndarray  # shape (N+1,)
    x: np.ndarray  # shape (J+1,)
    V: np.ndarray  # shape (N+1, J+1)
    pi: np.ndarray  # shape (N+1, J+1)
    iterations: np.ndarray  # policy iterations used at each time step, shape (N,)
    n_clipped: int  # nodes where pi left [0, pi_max] or V_xx >= 0 (expected 0)
    n_upwind: int  # nodes where the Peclet condition failed (expected 0)
    _interp: dict = field(init=False, repr=False)

    def __post_init__(self):
        self._interp = {
            "V": RegularGridInterpolator((self.t, self.x), self.V),
            "pi": RegularGridInterpolator((self.t, self.x), self.pi),
        }

    def value(self, t, x):
        return self._evaluate("V", t, x)

    def position(self, t, x):
        return self._evaluate("pi", t, x)

    def _evaluate(self, name, t, x):
        t, x = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(x, dtype=float))
        return self._interp[name](np.stack([t, x], axis=-1))


def solve_hjb(
    p: Params,
    mbar: float,
    nx: int = 800,
    nt: int = 400,
    max_iter: int = 20,
    tol: float = 1e-10,
    pi_max: float = 10.0,
) -> HJBSolution:
    """Solve V_t + sup_pi {[r x + pi (mu - r)] V_x + pi^2 sigma^2 V_xx / 2} = 0, V(T) = U(., mbar).

    Boundary values: merton_strategy_value at x_min, merton_value at x_max. max_iter=1 gives
    the lagged policy (one linear solve per step, using the policy of the step above).
    """
    x = np.linspace(p.x_min, p.x_max, nx + 1)
    t = np.linspace(0.0, p.T, nt + 1)
    dx = x[1] - x[0]

    V = np.empty((nt + 1, nx + 1))
    pi = np.empty((nt + 1, nx + 1))
    V[-1] = utility(x, mbar, p)
    pi[-1] = pi_terminal(x, mbar, p)
    iterations = np.zeros(nt, dtype=int)
    n_clipped = n_upwind = 0

    for n in range(nt - 1, -1, -1):
        dt = t[n + 1] - t[n]
        rhs = V[n + 1].copy()
        rhs[0] = merton_strategy_value(t[n], x[0], mbar, p)
        rhs[-1] = merton_value(t[n], x[-1], p)

        policy, previous = pi[n + 1], V[n + 1]
        for k in range(1, max_iter + 1):
            v, upwind = _solve_linear(policy, x, dx, dt, rhs, p)
            policy, clipped = _update_policy(v, dx, p, pi_max)
            iterations[n] = k
            converged = np.max(np.abs(v - previous)) < tol
            previous = v
            if converged:
                break

        V[n], pi[n] = v, policy
        n_clipped += clipped
        n_upwind += upwind

    return HJBSolution(t, x, V, pi, iterations, n_clipped, n_upwind)


def _solve_linear(policy, x, dx, dt, rhs, p: Params):
    """Solve (I - dt L) v = rhs for a fixed policy, with Dirichlet rows at both ends (Steps 4-5)."""
    alpha, beta, central = jump_rates(policy, x, dx, p)

    # Banded storage for solve_banded: row 0 the upper diagonal, row 1 the main, row 2 the lower.
    ab = np.zeros((3, x.size))
    ab[0, 1:] = -dt * beta[:-1]
    ab[1] = 1.0 + dt * (alpha + beta)
    ab[2, :-1] = -dt * alpha[1:]
    ab[1, 0], ab[0, 1] = 1.0, 0.0
    ab[1, -1], ab[2, -2] = 1.0, 0.0

    return solve_banded((1, 1), ab, rhs), int(np.sum(~central[1:-1]))


def _update_policy(v, dx, p: Params, pi_max):
    """pi = -(mu - r) V_x / (sigma^2 V_xx) at interior nodes, kept in [0, pi_max] (Step 6)."""
    v_x = (v[2:] - v[:-2]) / (2 * dx)
    v_xx = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx**2
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(v_xx < 0, -(p.mu - p.r) * v_x / (p.sigma**2 * v_xx), np.inf)
    clipped = int(np.sum((raw < 0) | (raw > pi_max)))
    interior = np.clip(raw, 0.0, pi_max)
    # The edge rows are Dirichlet, so pi isn't needed there; copy the neighbour for the FP solver.
    return np.concatenate([interior[:1], interior, interior[-1:]]), clipped
