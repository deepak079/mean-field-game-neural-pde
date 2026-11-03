"""Tests for the PINN equilibrium (src/mfg/pinn/equilibrium.py) with small networks at lam = 0,
where the equilibrium is the Merton mean."""

from mfg.benchmarks import population_mean
from mfg.params import Params
from mfg.pinn.equilibrium import (
    PinnEquilibriumSettings,
    Stage,
    pinn_best_response,
    solve_pinn_equilibrium,
)
from mfg.pinn.fokker_planck import DensityNetSettings, PositionNetSettings
from mfg.pinn.hjb import ValueNetSettings

P0 = Params().with_(lam=0.0)


def small_stage(steps):
    return Stage(
        value=ValueNetSettings(
            width=32,
            depth=3,
            n_inside=1000,
            n_kink=200,
            n_terminal=200,
            n_edge=50,
            adam_steps=steps,
            lbfgs_steps=100,
            device="cpu",
        ),
        position=PositionNetSettings(
            width=16, depth=2, n_points=5000, adam_steps=steps, lbfgs_steps=50, device="cpu"
        ),
        density=DensityNetSettings(
            width=32,
            depth=3,
            n_inside=1000,
            n_population=500,
            n_mass_times=8,
            n_mass_grid=201,
            adam_steps=steps // 2,
            lbfgs_steps=100,
            device="cpu",
        ),
    )


def test_warm_start_uses_copies_of_the_previous_networks():
    first = pinn_best_response(P0, 1.1, small_stage(0))
    second = pinn_best_response(P0, 1.1, small_stage(10), previous=first)
    assert second.value.net is not first.value.net
    assert second.density.net is not first.density.net
    assert second.position_net is not first.position_net


def test_lam_0_equilibrium_is_the_merton_mean():
    # Without the penalty Phi doesn't depend on m̄; the PINN Phi is the Merton mean up to
    # training error. About 40 s; observed: m̄ off by 1.3e-3, after 2 evaluations.
    settings = PinnEquilibriumSettings(first=small_stage(1500), warm=small_stage(300), tol=2e-3)
    eq = solve_pinn_equilibrium(P0, settings)
    assert abs(eq.mbar - population_mean(P0.T, P0)) < 5e-3
    assert len(eq.history) <= 4
