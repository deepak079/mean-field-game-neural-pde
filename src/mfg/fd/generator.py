"""The discrete generator L shared by the HJB and Fokker-Planck solvers.

(L V)_j = alpha_j V_{j-1} - (alpha_j + beta_j) V_j + beta_j V_{j+1}. Read as a Markov chain on
the grid: from node j an investor jumps down at rate alpha_j and up at rate beta_j. The HJB
uses L (backward), the Fokker-Planck equation its transpose (forward).
"""

import numpy as np

from mfg.params import Params


def jump_rates(policy, x, dx, p: Params):
    """alpha, beta >= 0 for the wealth SDE under `policy`, and where central V_x was used.

    Central differences where the Peclet condition s^2 >= |b| dx holds (second order), upwind
    elsewhere (always nonnegative rates). Appendix A, Step 4.
    """
    b = p.r * x + policy * (p.mu - p.r)
    s2 = (policy * p.sigma) ** 2
    diffusion = s2 / (2 * dx**2)
    central = s2 >= np.abs(b) * dx
    alpha = np.where(central, diffusion - b / (2 * dx), diffusion + np.maximum(-b, 0.0) / dx)
    beta = np.where(central, diffusion + b / (2 * dx), diffusion + np.maximum(b, 0.0) / dx)
    return alpha, beta, central
