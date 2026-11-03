import numpy as np
import pytest

from mfg.equilibrium import secant_fixed_point


def test_finds_the_fixed_point_of_a_contraction():
    # Phi(m) = 1 + 0.5 cos(m) has one fixed point; the secant reaches it in a few steps.
    fixed = secant_fixed_point(lambda m: (1 + 0.5 * np.cos(m), None), start=1.1)
    assert abs(1 + 0.5 * np.cos(fixed.mbar) - fixed.mbar) < 1e-10
    assert len(fixed.history) <= 8


def test_history_and_solution_follow_the_evaluations():
    calls = []

    def phi(m):
        calls.append(m)
        return 0.9 - 0.1 * m, f"solution at {m}"

    fixed = secant_fixed_point(phi, start=0.0)
    assert [m for m, _ in fixed.history] == calls
    assert fixed.history[1][0] == fixed.history[0][1]  # second point is Phi(start)
    assert fixed.solution == f"solution at {calls[-1]}"
    assert fixed.mbar == pytest.approx(0.9 / 1.1, abs=1e-12)  # linear g: exact after 3 calls
    assert len(calls) == 3


def test_a_constant_phi_needs_two_evaluations():
    fixed = secant_fixed_point(lambda m: (1.5, None), start=1.0)
    assert fixed.mbar == 1.5
    assert len(fixed.history) == 2


def test_gives_up_with_the_history_when_tol_is_below_the_noise():
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="no fixed point"):
        secant_fixed_point(
            lambda m: (1.0 + 1e-3 * rng.standard_normal(), None), 1.0, tol=1e-9, max_iter=5
        )
