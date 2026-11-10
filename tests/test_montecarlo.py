"""Tests for the Monte Carlo of the finite-N game (src/mfg/montecarlo.py): against the lam = 0
closed forms, and the seats and random numbers by hand."""

import numpy as np
import pytest

from mfg.benchmarks import merton_pi, population_mean, population_var
from mfg.montecarlo import crowding_base, group_statistics, mean_field_reference, simulate_games
from mfg.params import Params

P0 = Params().with_(lam=0.0)


def merton_rule(t, x, y):
    """Everyone at the Merton position, in the NumPy interface of NPlayerSolution.position."""
    return merton_pi(t, P0) * np.ones_like(x)


def test_crowding_base_by_hand():
    # 2 Phi(0.1 / (s_T sqrt(1 - 1/N))) - 1 with s_T = 0.363, in percent (the prompt's table).
    for N, expected in [(2, 30.3), (3, 26.4), (5, 24.2), (10, 22.8), (20, 22.3), (np.inf, 21.7)]:
        assert 100 * crowding_base(N, P0) == pytest.approx(expected, abs=0.1)


def test_merton_games_match_the_lam_0_population_and_crowding():
    # N = 3, lam = 0: every final wealth is N(M(T), S(T)^2) and independent, so the pooled mean and
    # variance are the closed forms, and the share within 0.1 of the group mean is base(3).
    games = simulate_games(merton_rule, P0, N=3, n_games=20_000, n_steps=100, seed=0)
    X = games.X_T
    assert X.shape == (20_000, 3) and games.n_clipped == 0
    stats = group_statistics(X, P0)
    n = X.size
    se_mean = np.sqrt(population_var(P0.T, P0) / n)
    assert stats["mean"] == pytest.approx(population_mean(P0.T, P0), abs=3 * se_mean)
    se_var = population_var(P0.T, P0) * np.sqrt(2 / (n - 1))
    assert X.var() == pytest.approx(population_var(P0.T, P0), abs=3 * se_var)
    # Games are the independent units for the share near the group mean.
    near = np.abs(X - X.mean(axis=1, keepdims=True)) < 0.1
    se_share = near.mean(axis=1).std() / np.sqrt(len(X))
    assert stats["within_band"] == pytest.approx(crowding_base(3, P0), abs=3 * se_share)
    assert stats["crowding_index"] == pytest.approx(1.0, abs=3 * se_share / crowding_base(3, P0))


def test_one_euler_step_by_hand_reads_each_investor_from_its_own_seat():
    # A rule that invests the mean wealth of one's rivals: with one step, X_T is explicit in the
    # starting wealths and the normal draws, which the generator gives in the same order.
    def rule(t, x, y):
        return y.mean(axis=-1) + 0 * x

    N, n_games, seed = 4, 5, 3
    games = simulate_games(rule, P0, N, n_games, n_steps=1, seed=seed)
    rng = np.random.default_rng(seed)
    X0 = rng.normal(P0.m0_mean, P0.m0_std, (n_games, N))
    Z = rng.standard_normal((n_games, N))
    pi = (X0.sum(axis=1, keepdims=True) - X0) / (N - 1)  # each investor: the others' mean
    dt = P0.T
    by_hand = X0 + (P0.r * X0 + pi * (P0.mu - P0.r)) * dt + P0.sigma * pi * np.sqrt(dt) * Z
    np.testing.assert_allclose(games.X_T, by_hand, rtol=1e-12)
    np.testing.assert_array_equal(games.X_0, X0)  # the starting wealths are kept (check C5)


def test_same_seed_gives_the_same_numbers_and_the_deviant_takes_seat_0():
    base = simulate_games(merton_rule, P0, N=3, n_games=200, n_steps=20, seed=5)
    again = simulate_games(merton_rule, P0, N=3, n_games=200, n_steps=20, seed=5)
    np.testing.assert_array_equal(base.X_T, again.X_T)
    # Investor 0 who never invests ends at x0 e^{rT} up to the Euler step, the others unchanged.
    deviant = simulate_games(
        merton_rule, P0, N=3, n_games=200, n_steps=20, seed=5, deviant=lambda t, x, y: 0 * x
    )
    np.testing.assert_array_equal(deviant.X_T[:, 1:], base.X_T[:, 1:])
    rng = np.random.default_rng(5)
    x0 = rng.normal(P0.m0_mean, P0.m0_std, (200, 3))[:, 0]
    np.testing.assert_allclose(deviant.X_T[:, 0], x0 * (1 + P0.r / 20) ** 20, rtol=1e-12)


def test_positions_are_clipped_and_counted():
    games = simulate_games(lambda t, x, y: -1.0 + 0 * x, P0, N=2, n_games=10, n_steps=3, seed=0)
    assert games.n_clipped == 10 * 2 * 3
    rng = np.random.default_rng(0)
    x0 = rng.normal(P0.m0_mean, P0.m0_std, (10, 2))
    np.testing.assert_allclose(games.X_T, x0 * (1 + P0.r / 3) ** 3, rtol=1e-12)  # pi = 0


def test_mean_field_reference_at_lam_0_is_the_gaussian():
    ref = mean_field_reference(P0, nx=200, nt=100)
    assert ref["mbar"] == pytest.approx(population_mean(P0.T, P0), abs=1e-3)
    assert ref["std"] == pytest.approx(np.sqrt(population_var(P0.T, P0)), rel=0.02)
    assert ref["crowding_index"] == pytest.approx(1.0, abs=0.02)
