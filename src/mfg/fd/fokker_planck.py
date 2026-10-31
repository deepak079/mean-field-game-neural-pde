"""Finite-difference solver for the Fokker-Planck equation under a given position pi(t, x).

The matrix is the transpose of the HJB's: the HJB generator L read as a Markov chain on the
grid, with no jumps off the ends (zero flux). Implicit Euler forward from m_0. The steps follow
Appendix B of docs/notes/problem21_hjb_fokker_planck.pdf.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_banded
from scipy.stats import norm

from mfg.fd.generator import jump_rates
from mfg.params import Params


@dataclass
class FPSolution:
    """The density on the grid (cell j is [x_j - dx/2, x_j + dx/2]), plus interpolation."""

    t: np.ndarray  # shape (N+1,)
    x: np.ndarray  # shape (J+1,)
    m: np.ndarray  # shape (N+1, J+1)
    n_upwind: int  # nodes where the Peclet condition failed (expected 0)
    _interp: RegularGridInterpolator = field(init=False, repr=False)

    def __post_init__(self):
        # Outside the grid there is no mass.
        self._interp = RegularGridInterpolator(
            (self.t, self.x), self.m, bounds_error=False, fill_value=0.0
        )

    @property
    def dx(self) -> float:
        return self.x[1] - self.x[0]

    @property
    def mass(self) -> np.ndarray:
        """Total mass at each time step (1 up to round-off)."""
        return self.dx * self.m.sum(axis=1)

    @property
    def mean(self) -> np.ndarray:
        """Average wealth m̄(t_n) at each time step; mean[-1] is m̄_T."""
        return self.dx * self.m @ self.x

    def density(self, t, x):
        t, x = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(x, dtype=float))
        return self._interp(np.stack([t, x], axis=-1))


def solve_fokker_planck(p: Params, position, nx: int = 800, nt: int = 400) -> FPSolution:
    """Solve m_t = -(b m)_x + (s^2 m)_xx / 2, m(0) = N(m0_mean, m0_std^2), zero flux at the ends.

    `position(t, x)` is the strategy everyone follows, e.g. HJBSolution.position. On the HJB's
    own grid (same nx, nt) the step from t_n to t_{n+1} uses exactly the HJB matrix at t_n.
    """
    x = np.linspace(p.x_min, p.x_max, nx + 1)
    t = np.linspace(0.0, p.T, nt + 1)
    dx = x[1] - x[0]

    m = np.empty((nt + 1, nx + 1))
    m[0] = _initial_density(x, dx, p)
    n_upwind = 0

    for n in range(nt):
        dt = t[n + 1] - t[n]
        alpha, beta, central = jump_rates(position(t[n], x), x, dx, p)
        alpha[0], beta[-1] = 0.0, 0.0  # zero flux: no jumps off the grid

        # Banded storage of (I - dt L)^T: its upper diagonal is L's lower one and vice versa.
        ab = np.zeros((3, nx + 1))
        ab[0, 1:] = -dt * alpha[1:]
        ab[1] = 1.0 + dt * (alpha + beta)
        ab[2, :-1] = -dt * beta[:-1]

        m[n + 1] = solve_banded((1, 1), ab, m[n])
        n_upwind += int(np.sum(~central))

    return FPSolution(t, x, m, n_upwind)


def _initial_density(x, dx, p: Params):
    """Cell averages of N(m0_mean, m0_std^2), scaled so the discrete mass is exactly 1."""
    cdf = norm.cdf(np.append(x - dx / 2, x[-1] + dx / 2), p.m0_mean, p.m0_std)
    cell_mass = np.diff(cdf)
    return cell_mass / (dx * cell_mass.sum())
