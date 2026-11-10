"""Monte Carlo of the finite-N game: play many games under a given rule and measure the group.

The N investors of a game start from independent N(m0_mean, m0_std^2) wealths and follow the rule
position(t, x, y), each from their own seat (the same `seats` function as the training, in
mfg.pinn.nplayer), by Euler steps. There is no Fokker-Planck equation in the N-player game, so
this is how the group's final mean, spread and crowding are read. The rule can be a trained
network (NPlayerSolution.position), an exact formula or, one day, a finite-difference solution.

The crowding index compares groups of different N fairly: the share of investors who end within
`band` of their group's mean, divided by the same share without the penalty (crowding_base), since
with few investors everyone is closer to a mean they count in. In the mean-field limit the index
uses the mass of the final density near the equilibrium mean (mean_field_reference).

This also covers the README task "Monte Carlo simulation of N investors under a given pi*".
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from mfg.benchmarks import population_var
from mfg.params import Params
from mfg.pinn.nplayer import seats


@dataclass
class GameResult:
    X_T: np.ndarray  # final wealths, shape (n_games, N); investor 0 is the deviant if there is one
    X_0: np.ndarray  # starting wealths, same shape (check C5 compares V(0, .) on them)
    n_clipped: int  # positions outside the clip range over all steps and investors (expected 0)
    seed: int
    n_steps: int


def simulate_games(position, p: Params, N, n_games, n_steps, seed, deviant=None, clip=(0.0, 10.0)):
    """Play n_games games of N investors who all follow `position` (NumPy: t scalar, x (n,),
    y (n, N - 1) -> pi (n,)); returns a GameResult.

    Euler: X <- X + (r X + pi (mu - r)) dt + sigma pi sqrt(dt) Z with dt = T / n_steps. With the
    same seed the same starting wealths and the same Z are drawn whatever the rule, so within one
    N the difference between two rules is not Monte Carlo noise. `deviant`: a rule for investor
    0 alone (the others keep `position`), to measure what ignoring the league costs.
    """
    rng = np.random.default_rng(seed)
    X_0 = rng.normal(p.m0_mean, p.m0_std, (n_games, N))
    X = X_0
    dt = p.T / n_steps
    n_clipped = 0
    for k in range(n_steps):
        t = k * dt
        own, others = seats(X)
        pi = position(t, own.ravel(), others.reshape(-1, N - 1)).reshape(n_games, N)
        if deviant is not None:
            pi[:, 0] = deviant(t, own[:, 0], others[:, 0])
        n_clipped += int(np.sum((pi < clip[0]) | (pi > clip[1])))
        pi = np.clip(pi, *clip)
        Z = rng.standard_normal((n_games, N))
        X = X + (p.r * X + pi * (p.mu - p.r)) * dt + p.sigma * pi * np.sqrt(dt) * Z
    return GameResult(X, X_0, n_clipped, seed, n_steps)


def crowding_base(N, p: Params, band=0.1):
    """Share of investors within `band` of their group's mean when nobody is penalised (lam = 0).

    Then the N final wealths are independent N(M_T, s_T^2), so X_i minus the group mean is
    N(0, s_T^2 (1 - 1/N)): the share is 2 Phi(band / (s_T sqrt(1 - 1/N))) - 1. N = inf gives the
    mean-field base, 2 Phi(band / s_T) - 1.
    """
    s_T = np.sqrt(population_var(p.T, p))
    return 2 * norm.cdf(band / (s_T * np.sqrt(1.0 - 1.0 / N))) - 1


def group_statistics(X_T, p: Params, band=0.1):
    """Final mean and spread of all the investors together, the share within `band` of their own
    group's mean, and the crowding index (that share over crowding_base)."""
    n_games, N = X_T.shape
    near = np.abs(X_T - X_T.mean(axis=1, keepdims=True)) < band
    within = float(near.mean())
    return {
        "N": int(N),
        "n_games": int(n_games),
        "mean": float(X_T.mean()),
        "std": float(X_T.std()),
        "within_band": within,
        "crowding_index": within / crowding_base(N, p, band),
        # Games are the independent units: the standard error of the mean uses them.
        "mean_stderr": float(X_T.mean(axis=1).std() / np.sqrt(n_games)),
    }


def mean_field_reference(p: Params, band=0.1, nx=800, nt=400):
    """The same numbers for infinitely many investors, from the 1-D finite-difference equilibrium
    (mfg.fd.equilibrium.solve_equilibrium, about 2 s): the equilibrium mean, the spread of the
    final density, its mass within `band` of the mean, and the crowding index with the N = inf
    base. Called with p.with_(lam=lambda_n(lam, N)) it gives the "my wealth moves the mean"
    effect alone, the reference curve of the analysis."""
    from mfg.fd.equilibrium import solve_equilibrium  # Deepak's solver, used as is

    eq = solve_equilibrium(p, nx, nt)
    x, m, dx = eq.fp.x, eq.fp.m[-1], eq.fp.dx
    mass = m.sum() * dx
    mean = (m * x).sum() * dx / mass
    std = np.sqrt((m * (x - mean) ** 2).sum() * dx / mass)
    within = m[np.abs(x - eq.mbar) < band].sum() * dx / mass
    return {
        "lam": float(p.lam),
        "mbar": float(eq.mbar),
        "std": float(std),
        "within_band": float(within),
        "crowding_index": float(within / crowding_base(np.inf, p, band)),
        "evaluations": len(eq.history),
    }
