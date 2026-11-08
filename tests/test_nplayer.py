"""Tests for the finite-N game (src/mfg/pinn/nplayer.py, SymmetricNet in nets.py and the C2 closed
form in benchmarks.py): exact formulas in float64, no training."""

import json
from dataclasses import asdict, replace
from functools import partial

import numpy as np
import pytest
import torch

from mfg.benchmarks import c2_k, c2_position, c2_value, lambda_n, merton_value
from mfg.params import Params
from mfg.pinn.nets import SymmetricNet
from mfg.pinn.nplayer import (
    C2Terminal,
    NetSettings,
    NPlayerSettings,
    NPlayerSolution,
    PointSettings,
    ShortfallTerminal,
    StageSettings,
    check_states,
    hjb_residual,
    is_accepted,
    merton_position,
    nplayer_loss,
    rival_positions,
    rival_seats,
    sample_batch,
    should_stop,
    solve_nplayer_equilibrium,
)
from mfg.pinn.nplayer_io import RunFolder, init_from, load_solution, save_checkpoint

P = Params()  # base parameters, lam = 2


# ---------------------------------------------------------------- the closed forms (tests 1, 2)


def test_lambda_n_by_hand():
    # lam (1 - 1/N)^2: 2 * 0.25 with N = 2 and 2 * 0.64 with N = 5.
    assert lambda_n(2.0, 2) == pytest.approx(0.5)
    assert lambda_n(2.0, 5) == pytest.approx(1.28)


def test_c2_numbers_by_hand():
    # Merton with risk aversion gamma (1 - eta/N) = 0.9: 1.4703 / 0.9 at t = 0.
    assert c2_position(0.0, 5, 0.5, P) == pytest.approx(1.6337, abs=5e-5)
    assert c2_k(5, 0.5) == pytest.approx(-5 / 162)
    # Without the comparison (eta = 0) C2 is Merton: k_N = -1/2 and the Merton value.
    assert c2_k(5, 0.0) == -0.5
    y = np.array([[0.8, 1.2, 1.0, 1.4]])
    assert c2_value(0.3, 1.1, y, 5, 0.0, P) == pytest.approx(merton_value(0.3, 1.1, P))


# ---------------------------------------------------------------- the symmetric network (test 7)


def test_symmetric_net_ignores_the_order_of_the_rivals_and_works_for_any_n():
    torch.manual_seed(0)
    net = SymmetricNet(P).double()
    t = torch.full((3, 1), 0.4, dtype=torch.float64)
    x = torch.tensor([[0.9], [1.1], [1.6]], dtype=torch.float64)
    y = torch.tensor([[1.2, 0.9, 1.0, 1.4], [0.5, 1.0, 1.5, 2.0], [1.0, 1.0, 0.8, 1.3]]).double()
    v = net(t, x, y)
    assert v.shape == (3, 1)
    shuffled = net(t, x, y[:, [2, 0, 3, 1]])
    torch.testing.assert_close(shuffled, v, atol=1e-6, rtol=0.0)
    # The same weights serve any number of rivals (the summaries are averaged, not summed).
    assert net(t, x, y[:, :1]).shape == (3, 1)  # N = 2
    assert net(t, x, torch.cat([y, y, y[:, :1]], dim=1)).shape == (3, 1)  # N = 10
    # Summary net 1 -> 32 -> 32 -> 32 (2176 weights) and final net 35 -> 4 x 64 -> 1 (14849).
    assert sum(w.numel() for w in net.parameters()) == 17025


# ---------------------------------------------------------------- seats, rivals, residual (3-6, 8)


def columns(t, x, y):
    """Column tensors in float64 that track gradients, as the PINN functions expect."""

    def as_tensor(a):
        return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float64).requires_grad_()

    return as_tensor(t).reshape(-1, 1), as_tensor(x).reshape(-1, 1), as_tensor(y)


def random_states(N, n=40, seed=0):
    """Random (t, x, y) around the population, in float64 columns."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 0.9, n)
    w = rng.normal(1.0, 0.3, (n, N))
    return columns(t, w[:, 0], w[:, 1:])


def c2_v(N, eta):
    """The exact C2 solution v = log(-V) in torch (benchmarks.c2_value), of (t, x, y)."""
    a, b = 1.0 - eta / N, eta / N

    def v(t, x, y):
        tau = P.T - t
        linear = a * x - b * y.sum(dim=1, keepdim=True)
        return -P.gamma * torch.exp(P.r * tau) * linear + P.kappa**2 * c2_k(N, eta) * tau

    return v


def merton_v(t, x, y):
    """Exact v without the penalty (lam = 0); it does not depend on the rivals."""
    tau = P.T - t
    merton = -P.gamma * x * torch.exp(P.r * tau) - 0.5 * P.kappa**2 * tau
    return merton + 0.0 * y.sum(1, keepdim=True)


def test_seats_of_the_hand_example():
    # N = 4, x = 1, y = (2, 3, 4): the rival with 3 sees itself with 3 and its rivals {1, 2, 4}.
    x, y = np.array([[1.0]]), np.array([[2.0, 3.0, 4.0]])
    own, others = rival_seats(x, y)
    assert own.shape == (1, 3) and others.shape == (1, 3, 3)
    assert own[0, 1] == 3.0
    assert set(others[0, 1]) == {1.0, 2.0, 4.0}
    # Every seat: its own wealth is not among its rivals, and my wealth always is.
    for j in range(3):
        assert own[0, j] not in others[0, j]
        assert 1.0 in others[0, j]
    # The same on tensors.
    own_t, others_t = rival_seats(torch.tensor(x), torch.tensor(y))
    assert torch.equal(own_t, torch.tensor(own)) and torch.equal(others_t, torch.tensor(others))


def test_rival_positions_of_a_made_up_rule_detect_a_badly_built_seat():
    # v = -own + 0.1 own ybar gives pi = 1.5 / (1 - 0.1 ybar) at a seat. For the rival with 3 in
    # the hand example, ybar = (1 + 2 + 4) / 3 = 7/3 and pi = 1.95652; a seat that forgets my
    # wealth or keeps its own gives 2.142857, one with all four wealths 2.0.
    def fake_v(t, own, others):
        return -own + 0.1 * own * others.mean(dim=1, keepdim=True)

    t, x, y = columns([0.5], [1.0], [[2.0, 3.0, 4.0]])
    pi, n_floor, n_cap = rival_positions(fake_v, t, x, y, P)
    np.testing.assert_allclose(pi.numpy(), [[1.5 / (1 - 0.8 / 3), 1.95652, 1.875]], atol=1e-5)
    assert (n_floor, n_cap) == (0, 0)
    assert not pi.requires_grad


@pytest.mark.parametrize("N", [3, 5])
def test_c2_solution_has_zero_residual_with_the_rivals_in_their_rule(N):
    t, x, y = random_states(N)
    pi_rivals, _, _ = rival_positions(c2_v(N, 0.5), t, x, y, P)
    residual = hjb_residual(c2_v(N, 0.5), pi_rivals, t, x, y, P)
    assert residual.abs().max().item() < 1e-10


def test_merton_solution_has_zero_residual_whatever_the_rivals_do():
    # Without the penalty v does not depend on y, so the rivals' terms vanish for any pi_j.
    t, x, y = random_states(4)
    pi_rivals = torch.rand(len(t), 3, dtype=torch.float64) * 3
    assert hjb_residual(merton_v, pi_rivals, t, x, y, P).abs().max().item() < 1e-10


def test_rival_positions_of_the_c2_solution_are_its_exact_rule():
    N, eta = 5, 0.5
    t, x, y = random_states(N)
    pi, n_floor, n_cap = rival_positions(c2_v(N, eta), t, x, y, P)
    assert pi.shape == (len(t), N - 1)
    exact = c2_position(t.detach().numpy(), N, eta, P)  # (n, 1), the same for every rival
    np.testing.assert_allclose(pi.numpy(), np.repeat(exact, N - 1, axis=1), rtol=1e-10)
    assert (n_floor, n_cap) == (0, 0)


def test_rival_positions_count_the_clips():
    # A rule that wants a negative position and one that wants more than the cap.
    t, x, y = columns([0.5], [1.0], [[2.0, 3.0, 4.0]])
    pi, n_floor, _ = rival_positions(
        lambda t, own, others: own + 0 * others.sum(1, keepdim=True), t, x, y, P
    )
    assert n_floor == 3 and torch.all(pi == 0.0)
    # v = log(4.5 - own) has v_xx + v_x^2 = 0, so the curvature floor acts and the position wanted
    # is 30 / (4.5 - own) = 12, 20 and 60 at the three seats: all above the cap.
    def flat_v(t, own, others):
        return torch.log(4.5 - own) + 0 * others.sum(1, keepdim=True)

    pi, _, n_cap = rival_positions(flat_v, t, x, y, P)
    assert n_cap == 3 and torch.all(pi == 10.0)


@pytest.mark.parametrize("N", [2, 5])
def test_terminal_condition_equals_the_formula_with_the_group_mean(N):
    rng = np.random.default_rng(1)
    x = rng.normal(1.0, 0.4, (50, 1))
    y = rng.normal(1.0, 0.4, (50, N - 1))
    v_T, v_T_x, v_T_xx, away = ShortfallTerminal(P, N).targets(x, y)
    xbar = (x + y.sum(axis=1, keepdims=True)) / N
    by_hand = np.log(np.exp(-P.gamma * x) + 0.5 * P.lam * np.maximum(xbar - x, 0.0) ** 2)
    np.testing.assert_allclose(v_T, by_hand, rtol=1e-12)
    # Above the rivals' mean the slope is -gamma and the curvature 0, as in 1-D.
    above = (x > y.mean(axis=1, keepdims=True)).ravel()
    np.testing.assert_allclose(v_T_x[above], -P.gamma)
    np.testing.assert_allclose(v_T_xx[above], 0.0, atol=1e-12)
    assert away.shape == (50, 1) and away.dtype == bool


# ---------------------------------------------------------------- points, check states, loss (9)

SMALL_POINTS = PointSettings(n_group=50, n_far=40, n_kink=30, n_terminal=20)


def merton_rivals(t, x, y):
    """Every rival at the Merton position, for a batch on the CPU."""
    return merton_position(t, P) * torch.ones_like(y)


def test_sampled_points_have_the_right_shapes_and_lie_where_they_should():
    N = 4
    rng = np.random.default_rng(0)
    batch = sample_batch(P, N, SMALL_POINTS, rng, "cpu", ShortfallTerminal(P, N), merton_rivals)
    n_eq = 50 + 40 + 30
    assert batch["t"].shape == (n_eq, 1) and batch["x"].shape == (n_eq, 1)
    assert batch["y"].shape == (n_eq, N - 1) and batch["pi_rivals"].shape == (n_eq, N - 1)
    assert batch["t_T"].shape == (20, 1) and torch.all(batch["t_T"] == P.T)
    assert batch["y_T"].shape == (20, N - 1)
    for key in ("v_T", "v_T_x", "v_T_xx", "away"):
        assert batch[key].shape == (20, 1)
    # The kink points come last: close to T, and my wealth close to the rivals' mean.
    t_kink, x_kink, y_kink = (batch[k][-30:] for k in ("t", "x", "y"))
    assert torch.all(t_kink >= P.T - 0.1 - 1e-6)
    assert torch.all((x_kink - y_kink.mean(dim=1, keepdim=True)).abs() <= 0.3 + 1e-6)
    # Group points: the N wealths of a point stay together (one group, one centre and width).
    w = torch.cat([batch["x"][:50], batch["y"][:50]], dim=1)
    assert torch.all(w.std(dim=1) < 1.0)
    # `away` marks the terminal points further than the band from the rivals' mean.
    gap = (batch["x_T"] - batch["y_T"].mean(dim=1, keepdim=True)).abs()
    assert torch.equal(batch["away"], gap > 0.05)
    assert torch.all(batch["pi_rivals"] > 1.4)  # Merton, 1.47 to 1.5


def test_check_states_repeat_with_the_seed_and_have_the_right_shapes():
    first = check_states(P, 5, n=300, seed=1, t_max=0.9)
    again = check_states(P, 5, n=300, seed=1, t_max=0.9)
    assert first["t"].shape == (300,) and first["x"].shape == (300,)
    assert first["y"].shape == (300, 4)
    assert first["t"].min() >= 0.0 and first["t"].max() <= 0.9
    for key in ("t", "x", "y"):
        np.testing.assert_array_equal(first[key], again[key])
    assert not np.array_equal(first["x"], check_states(P, 5, n=300, seed=2, t_max=0.9)["x"])


def test_loss_vanishes_for_the_exact_c2_solution():
    N, eta = 3, 0.5
    rng = np.random.default_rng(0)

    def rivals(t, x, y):
        return rival_positions(c2_v(N, eta), t, x, y, P)[0]

    batch = sample_batch(
        P, N, SMALL_POINTS, rng, "cpu", C2Terminal(P, N, eta), rivals, dtype=torch.float64
    )
    total, parts = nplayer_loss(c2_v(N, eta), batch, P, NPlayerSettings())
    assert set(parts) == {"equation", "terminal", "slopes"}
    assert total.item() < 1e-20


def test_loss_of_a_fresh_network_is_finite_and_reaches_every_weight():
    N = 3
    torch.manual_seed(0)
    net = SymmetricNet(P, features=8, summary_width=8, width=16, depth=2)
    rng = np.random.default_rng(0)
    batch = sample_batch(P, N, SMALL_POINTS, rng, "cpu", ShortfallTerminal(P, N), merton_rivals)
    total, parts = nplayer_loss(net, batch, P, NPlayerSettings())
    assert np.isfinite(total.item()) and all(np.isfinite(v.item()) for v in parts.values())
    total.backward()
    assert all(w.grad is not None and torch.isfinite(w.grad).all() for w in net.parameters())


# ---------------------------------------------------------------- the solution, rounds, saving (11)

TINY = dict(
    net=NetSettings(features=4, summary_width=8, width=8, depth=2),
    points=PointSettings(n_group=20, n_far=20, n_kink=10, n_terminal=10),
    first_round=StageSettings(adam_steps=2, lbfgs_steps=2),
    later_rounds=StageSettings(adam_steps=2, lbfgs_steps=0),
    n_check=50,
    device="cpu",
)


def tiny_solution(N, seed=0):
    torch.manual_seed(seed)
    return NPlayerSolution(SymmetricNet(P, **asdict(TINY["net"])), P, N)


def test_solution_broadcasts_like_t_plus_x_with_the_rivals_on_the_last_axis():
    sol = tiny_solution(N=4)
    x = np.linspace(0.0, 2.0, 5)
    y = np.array([0.9, 1.1, 1.3])
    assert sol.value(0.5, x, y).shape == (5,)  # one time, five wealths, one group of rivals
    assert sol.position(np.zeros((3, 1)), x, y).shape == (3, 5)  # three times by five wealths
    assert sol.value(0.5, 1.0, np.ones((7, 3))).shape == (7,)  # seven groups of rivals
    with pytest.raises(ValueError):
        sol.value(0.5, 1.0, np.ones(2))  # wrong number of rivals


def test_solution_position_matches_finite_differences_of_its_value():
    # pi* = -(mu - r)/sigma^2 V_x / V_xx for any smooth V, so an untrained network will do. The
    # network works in float32, so the differences need a wide step (observed error 3e-4).
    sol = tiny_solution(N=3)
    t, x, y, h = 0.4, np.array([0.8, 1.0, 1.5]), np.array([1.1, 0.9]), 0.05
    V = sol.value(t, x, y)
    V_x = (sol.value(t, x + h, y) - sol.value(t, x - h, y)) / (2 * h)
    V_xx = (sol.value(t, x + h, y) - 2 * V + sol.value(t, x - h, y)) / h**2
    np.testing.assert_allclose(sol.position(t, x, y), -1.5 * V_x / V_xx, rtol=5e-3)


def test_stop_rule_and_acceptance():
    # From round 3, stop as soon as the change no longer halves; accept if the last one is small.
    assert not should_stop([0.5, 0.1], min_rounds=3)
    assert not should_stop([0.5, 0.1, 0.04], min_rounds=3)  # still halving
    assert should_stop([0.5, 0.1, 0.06], min_rounds=3)
    assert is_accepted([0.5, 0.1, 0.06], min_rounds=3, accept_delta=0.02) is False
    assert is_accepted([0.5, 0.02, 0.015], min_rounds=3, accept_delta=0.02) is True
    assert is_accepted([0.5, 0.1, 0.04], min_rounds=3, accept_delta=0.02) is False  # not stopped


def test_rounds_run_save_resume_and_reload(tmp_path):
    # Two rounds on the CPU with a tiny network: the mechanics only, no accuracy.
    N = 2
    s = NPlayerSettings(**TINY, max_rounds=2, min_rounds=2)
    folder = RunFolder(tmp_path, N, seed=0)
    on_round = partial(folder.save_round, p=P, N=N, settings=s)

    # Interrupted on purpose after round 1 ...
    first = solve_nplayer_equilibrium(P, N, s, on_round=on_round, stop_after=1)
    assert [r.round for r in first.rounds] == [1] and not first.finished
    assert folder.rounds_saved() == [1] and folder.status()["status"] == "running"
    # ... and resumed from the saved round: the next round is number 2, with the saved rule as
    # the rule before.
    start = folder.start_point("cpu")
    assert start.rounds_done == 1 and len(start.deltas) == 1 and start.pi_before.shape == (50,)
    second = solve_nplayer_equilibrium(P, N, s, start=start, on_round=on_round)
    assert [r.round for r in second.rounds] == [2] and second.finished
    assert second.deltas == pytest.approx([*first.deltas, second.rounds[0].delta])
    folder.finish(second)
    assert folder.status()["status"] in ("accepted", "not_accepted")
    assert folder.rounds_saved() == [1, 2]
    history = json.loads(folder.history_path.read_text())
    assert [h["round"] for h in history] == [1, 2]
    assert history[0]["guard_winner"] in ("adam", "lbfgs") and history[1]["guard_winner"] == "none"
    saved = np.load(folder.pi_check_path)
    assert saved["pi"].shape == (2, 50)
    np.testing.assert_array_equal(saved["pi"][1], second.rounds[0].pi_check)

    # The saved network gives the same answers as the one in memory, from a plain-types file.
    loaded = load_solution(folder.final_path, device="cpu")
    x, y = np.linspace(0.0, 2.0, 5), np.array([1.1])
    np.testing.assert_allclose(loaded.value(0.5, x, y), second.solution.value(0.5, x, y), rtol=1e-6)
    np.testing.assert_allclose(
        loaded.position(0.5, x, y), second.solution.position(0.5, x, y), rtol=1e-4
    )
    raw = torch.load(folder.final_path, weights_only=True)
    assert raw["N"] == 2 and raw["round"] == 2 and raw["lam_N"] == pytest.approx(0.5)
    assert raw["params"]["lam"] == P.lam and "state_dict" in raw and "settings" in raw


def test_init_from_loads_the_weights_that_fit_and_warns_about_the_rest(tmp_path):
    N = 3
    s = NPlayerSettings(**TINY)
    source = tiny_solution(N, seed=1)
    path = tmp_path / "source.pt"
    save_checkpoint(path, source.net, P, N, round_=1, delta=0.1, settings=s)
    # Same architecture: everything fits, and N may differ (the same weights serve any N).
    same = init_from(path, P, s, device="cpu")
    for a, b in zip(same.parameters(), source.net.parameters(), strict=True):
        assert torch.equal(a, b)
    # A wider final network: the summary net fits, the final one only partly.
    wider = replace(s, net=replace(s.net, width=12))
    with pytest.warns(UserWarning, match="did not fit"):
        net = init_from(path, P, wider, device="cpu")
    assert torch.equal(net.summary[0].weight, source.net.summary[0].weight)
    assert net.final[0].weight.shape == (12, 7)
