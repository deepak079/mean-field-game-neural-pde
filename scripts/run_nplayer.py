"""Train the networks of the finite-N game, N by N and round by round, from configs/nplayer.yaml.

Everything is saved as it goes (results/nplayer/<run_name>/, see src/mfg/pinn/nplayer_io.py), so
a relaunch skips what is finished and continues what was interrupted from its last saved round.

Usage, from the repo folder with PYTHONPATH=src:
    python scripts/run_nplayer.py --plan                  print what is done and what is missing
    python scripts/run_nplayer.py                         train everything missing in the config
    python scripts/run_nplayer.py --N 5 --seed 0          one N and one seed only
    python scripts/run_nplayer.py --check c2              the exact check C2 (N = 5, about 10 min)
    python scripts/run_nplayer.py --check c1              optional: lam = 0 against Merton
    python scripts/run_nplayer.py --N 20 --init-from results/nplayer/main/N10_seed0/final.pt
    python scripts/run_nplayer.py --time-steps 100        seconds and memory per Adam step, per N
    python scripts/run_nplayer.py --fresh-best-response --N 5    optional check C6
    python scripts/run_nplayer.py --smoke                 tiny end-to-end run (under 2 minutes)
    python scripts/run_nplayer.py --force ...             redo even what is finished
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, replace
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from mfg.benchmarks import c2_position, c2_value, merton_pi, merton_value
from mfg.params import Params
from mfg.pinn.nets import SymmetricNet, pick_device
from mfg.pinn.nplayer import (
    C2Terminal,
    FrozenRivals,
    NPlayerSolution,
    RoundsStart,
    ShortfallTerminal,
    check_states,
    nplayer_loss,
    sample_batch,
    settings_from_dict,
    solve_nplayer_equilibrium,
    train_round,
)
from mfg.pinn.nplayer_io import RunFolder, git_commit, init_from, load_solution

RESULTS = Path("results/nplayer")
FINISHED = ("accepted", "not_accepted")

# A harmless notice PyTorch prints the first time it differentiates on a GPU.
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")


# ---------------------------------------------------------------- configuration


def load_config(path):
    """The config dict and the Params it describes (base.yaml with its lam replaced)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = Params.from_yaml(cfg["base"]).with_(lam=float(cfg["lam"]))
    return cfg, p


def settings_for(cfg, N, seed, device=None):
    """NPlayerSettings of one (N, seed): the config's settings, the points scaled for this N
    (plan B) and the seed."""
    s = settings_from_dict(cfg.get("settings") or {})
    scale = cfg.get("points_scale") or {}
    if N in scale:
        s = replace(s, points=s.points.scaled(scale[N]))
    return replace(s, seed=seed, device=device or s.device)


def run_root(cfg):
    return RESULTS / cfg["run_name"]


def jobs(cfg, args):
    """The (N, seed) pairs to train: the config's, or the ones asked for with --N / --seed."""
    if args.N is not None:
        seeds = [args.seed] if args.seed is not None else list(cfg["seeds"])
        return [(args.N, seed) for seed in seeds]
    pairs = [(N, seed) for N in cfg["N"] for seed in cfg["seeds"]]
    for N, extra in (cfg.get("extra_seeds") or {}).items():
        pairs += [(int(N), seed) for seed in extra]
    if args.seed is not None:
        pairs = [(N, seed) for N, seed in pairs if seed == args.seed]
    return pairs


def write_run_info(root, cfg, p):
    """config.yaml (with the Params) and run_info.json (commit, date, torch, GPU, commands)."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({**cfg, "params": asdict(p)}, f, sort_keys=False)
    path = root / "run_info.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"commands": []}
    info.update(
        git_commit=git_commit(),
        date=datetime.now().isoformat(timespec="seconds"),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    )
    info["commands"].append(" ".join(sys.argv))
    path.write_text(json.dumps(info, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- the plan and one job


def describe(folder):
    """One line on the state of a RunFolder."""
    st = folder.status()
    if st is None:
        return "missing"
    deltas = ", ".join(f"{d:.4f}" for d in st.get("deltas", []))
    if st["status"] in FINISHED:
        return f"done ({st['status']}), {st['rounds_done']} rounds, deltas {deltas}"
    return f"{st['status']}, {st['rounds_done']} rounds saved: will resume"


def plan(cfg, root, pairs):
    print(f"Run '{cfg['run_name']}' in {root}   lam = {cfg['lam']}")
    for N, seed in pairs:
        print(f"  N = {N:2d}, seed {seed}: {describe(RunFolder(root, N, seed))}")


def train_job(
    p, N, seed, settings, root, force=False, init_path=None, terminal=None, stop_after=None
):
    """Train one (N, seed) into root, resuming or skipping as its status says; returns the
    NPlayerEquilibrium, or None if it was already finished."""
    folder = RunFolder(root, N, seed)
    if force:
        folder.clear()
    status = folder.status()
    if status is not None and status["status"] in FINISHED:
        print(f"N = {N}, seed {seed}: already {status['status']}, skipped (use --force to redo)")
        return None

    device = settings.device or pick_device()
    start = folder.start_point(device)
    if start is not None:
        print(f"N = {N}, seed {seed}: resuming after round {start.rounds_done}")
    elif init_path is not None:
        print(f"N = {N}, seed {seed}: starting from {init_path}")
        start = RoundsStart(net=init_from(init_path, p, settings, device))
    states = check_states(p, N, settings.n_check, settings.check_seed, settings.check_t_max)
    on_round = partial(folder.save_round, p=p, N=N, settings=settings, states=states)
    folder.write_status("running")
    try:
        eq = solve_nplayer_equilibrium(
            p, N, settings, terminal=terminal, start=start, on_round=on_round, stop_after=stop_after
        )
    except Exception as error:  # noqa: BLE001 - record the failure, then let it show
        folder.write_status("failed", error=repr(error))
        raise
    if eq.finished:
        folder.finish(eq)
    deltas = ", ".join(f"{d:.4f}" for d in eq.deltas)
    verdict = "accepted" if eq.accepted else ("interrupted" if not eq.finished else "NOT accepted")
    print(
        f"N = {N}, seed {seed}: {len(eq.deltas)} rounds, deltas {deltas}: {verdict}"
        f"  ({eq.seconds / 60:.1f} min)"
    )
    return eq


def train_all(cfg, p, args, root):
    write_run_info(root, cfg, p)
    for N, seed in jobs(cfg, args):
        settings = settings_for(cfg, N, seed)
        train_job(p, N, seed, settings, root, force=args.force, init_path=args.init_from)


# ---------------------------------------------------------------- the exact checks C1 and C2


def errors_on_check_states(solution, settings, exact_position, exact_value):
    """Mean and max |pi - exact| and the max relative error of V on the check states."""
    s = check_states(
        solution.p, solution.N, settings.n_check, settings.check_seed, settings.check_t_max
    )
    pi = solution.position(s["t"], s["x"], s["y"])
    V = solution.value(s["t"], s["x"], s["y"])
    pi_error = np.abs(pi - exact_position(s["t"], s["x"], s["y"]))
    V_exact = exact_value(s["t"], s["x"], s["y"])
    return {
        "pi_mean_error": float(pi_error.mean()),
        "pi_max_error": float(pi_error.max()),
        "V_max_relative_error": float(np.max(np.abs(V - V_exact) / np.abs(V_exact))),
        "pi_at_t0": float(np.mean(pi[s["t"] < 0.05])) if np.any(s["t"] < 0.05) else float("nan"),
    }


def check_c2(cfg, p, args):
    """Check C2: the original utility -exp(-gamma (x - eta xbar)), whose equilibrium is exact.

    Everyone plays c2_position (Merton with risk aversion gamma (1 - eta/N)), so the rule should
    not change after round 1 and the rounds should stop at 3 or 4. Good if the mean error of pi
    is below 0.02, the max below 0.07 and V within 0.5%, on the check states.
    """
    conf = cfg["checks"]["c2"]
    N, eta, seed = int(conf["N"]), float(conf["eta"]), cfg["seeds"][0]
    root = RESULTS / "checks" / "c2"
    settings = settings_for(cfg, N, seed)
    write_run_info(root, cfg, p)
    train_job(p, N, seed, settings, root, force=args.force, terminal=C2Terminal(p, N, eta))
    folder = RunFolder(root, N, seed)
    solution = load_solution(folder.final_path)
    errors = errors_on_check_states(
        solution,
        settings,
        lambda t, x, y: c2_position(t, N, eta, p),
        lambda t, x, y: c2_value(t, x, y, N, eta, p),
    )
    rounds = folder.status()["rounds_done"]
    errors.update(
        N=N,
        eta=eta,
        rounds=rounds,
        deltas=folder.status().get("deltas"),
        exact_pi_at_t0=float(c2_position(0.0, N, eta, p)),
        passed=bool(
            errors["pi_mean_error"] < 0.02
            and errors["pi_max_error"] < 0.07
            and errors["V_max_relative_error"] < 0.005
            and rounds in (3, 4)
        ),
    )
    (folder.folder / "c2_errors.json").write_text(json.dumps(errors, indent=1), encoding="utf-8")
    print(json.dumps(errors, indent=1))
    print("C2 check:", "PASSED" if errors["passed"] else "NOT passed (see the numbers above)")
    return errors


def check_c1(cfg, p, args):
    """Optional check C1: without the penalty (lam = 0) the equilibrium is Merton for any N."""
    N, seed = int(cfg["checks"]["c1"]["N"]), cfg["seeds"][0]
    p0 = p.with_(lam=0.0)
    root = RESULTS / "checks" / "c1"
    settings = settings_for(cfg, N, seed)
    write_run_info(root, cfg, p0)
    train_job(p0, N, seed, settings, root, force=args.force)
    folder = RunFolder(root, N, seed)
    solution = load_solution(folder.final_path)
    errors = errors_on_check_states(
        solution,
        settings,
        lambda t, x, y: merton_pi(t, p0),
        lambda t, x, y: merton_value(t, x, p0),
    )
    errors.update(N=N, rounds=folder.status()["rounds_done"], deltas=folder.status().get("deltas"))
    (folder.folder / "c1_errors.json").write_text(json.dumps(errors, indent=1), encoding="utf-8")
    print(json.dumps(errors, indent=1))
    return errors


# ---------------------------------------------------------------- timing and C6


def time_steps(cfg, p, args, root):
    """Seconds and memory of one Adam step for every N of the config, with frozen-network
    rivals (the expensive case), and the rough time of a full training from them."""
    device = pick_device()
    rows = []
    for N in cfg["N"]:
        s = settings_for(cfg, N, cfg["seeds"][0], device)
        torch.manual_seed(0)
        net = SymmetricNet(p, **asdict(s.net)).to(device)
        rivals = FrozenRivals(net, p, s.rival_clip)
        terminal = ShortfallTerminal(p, N, s.points.kink_band)
        rng = np.random.default_rng(0)
        adam = torch.optim.Adam(net.parameters(), lr=1e-3)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Defaults bind this iteration's objects (ruff B023); step() is only called within it.
        def step(N=N, s=s, net=net, rivals=rivals, terminal=terminal, rng=rng, adam=adam):
            batch = sample_batch(p, N, s.points, rng, device, terminal, rivals)
            total, _ = nplayer_loss(net, batch, p, s)
            adam.zero_grad()
            total.backward()
            adam.step()

        def sync():
            if device == "cuda":
                torch.cuda.synchronize()

        for _ in range(3):  # warm-up
            step()
        sync()
        start = time.time()
        for _ in range(args.time_steps):
            step()
        sync()
        per_step = (time.time() - start) / args.time_steps
        batch = sample_batch(p, N, s.points, rng, device, terminal, rivals)
        start = time.time()
        for _ in range(5):
            rivals(batch["t"], batch["x"], batch["y"])
        sync()
        rivals_share = (time.time() - start) / 5 / per_step
        memory = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
        # L-BFGS iterations cost about two loss evaluations each; rounds assumed: 1 full + 3 warm.
        first = s.first_round.adam_steps + 2 * s.first_round.lbfgs_steps
        later = s.later_rounds.adam_steps + 2 * s.later_rounds.lbfgs_steps
        rows.append(
            {
                "N": N,
                "seconds_per_step": per_step,
                "rivals_share": rivals_share,
                "peak_gpu_memory_gb": memory,
                "estimated_minutes_4_rounds": per_step * (first + 3 * later) / 60,
            }
        )
        print(
            f"N = {N:2d}: {1000 * per_step:6.0f} ms per step, rivals {100 * rivals_share:3.0f}%, "
            f"peak memory {memory:.2f} GB, "
            f"about {rows[-1]['estimated_minutes_4_rounds']:.0f} min for 4 rounds"
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "timing.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    slow = [r["N"] for r in rows if r["seconds_per_step"] > 0.5]
    if slow:
        print(
            f"Plan B: a step takes over 0.5 s for N = {slow}; "
            f"set points_scale in the config, e.g. {{{slow[-1]}: 0.5}}"
        )
    else:
        print("No plan B needed: every step is under 0.5 s.")


def fresh_best_response(cfg, p, args, root):
    """Optional check C6: train from scratch the best response to the final rivals of a finished
    (N, seed), and compare its rule with the final one on the check states. If the difference is
    below about twice the spread between two seeds, the final rule is an equilibrium to the
    precision of the method."""
    for N, seed in jobs(cfg, args):
        folder = RunFolder(root, N, seed)
        if not folder.final_path.exists():
            print(f"N = {N}, seed {seed}: nothing trained yet, skipped")
            continue
        s = settings_for(cfg, N, seed)
        device = s.device or pick_device()
        final = load_solution(folder.final_path, device)
        torch.manual_seed(s.seed + 100)
        fresh = SymmetricNet(p, **asdict(s.net)).to(device)
        rivals = FrozenRivals(final.net, p, s.rival_clip)
        terminal = ShortfallTerminal(p, N, s.points.kink_band)
        training = train_round(fresh, rivals, p, N, s, s.first_round, terminal, seed=s.seed + 100)
        states = check_states(p, N, s.n_check, s.check_seed, s.check_t_max)
        pi_final = final.position(states["t"], states["x"], states["y"])
        pi_fresh = NPlayerSolution(fresh, p, N).position(states["t"], states["x"], states["y"])
        result = {
            "N": N,
            "seed": seed,
            "mean_abs_difference": float(np.mean(np.abs(pi_fresh - pi_final))),
            "max_abs_difference": float(np.max(np.abs(pi_fresh - pi_final))),
            "seconds": training.seconds,
            "guard_winner": training.guard_winner,
        }
        path = folder.folder / "c6_fresh_best_response.json"
        path.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result, indent=1))


# ---------------------------------------------------------------- the smoke test


def smoke_config(cfg):
    """A tiny copy of the config: N = 2 and 3, a few steps, few points, a small Monte Carlo."""
    tiny = dict(cfg)
    tiny.update(
        run_name="smoke",
        N=[2, 3],
        seeds=[0],
        extra_seeds={},
        points_scale={},
        settings={
            **(cfg.get("settings") or {}),
            "max_rounds": 3,
            "min_rounds": 2,
            "n_check": 500,
            "points": {
                **cfg["settings"]["points"],
                "n_group": 200,
                "n_far": 200,
                "n_kink": 200,
                "n_terminal": 200,
            },
            "first_round": {"adam_steps": 40, "lr_start": 1e-3, "lr_end": 1e-4, "lbfgs_steps": 5},
            "later_rounds": {"adam_steps": 20, "lr_start": 3e-4, "lr_end": 1e-4, "lbfgs_steps": 5},
            "net": {"features": 8, "summary_width": 8, "width": 16, "depth": 2},
            "print_every": 0,
        },
        montecarlo={**cfg["montecarlo"], "investors_per_N": 2000, "n_steps": 50},
        analysis={"fd_nx": 200, "fd_nt": 100},
    )
    return tiny


def smoke(cfg, p, args):
    """Run everything once with tiny settings: train, save, resume, load, --init-from, analyse.
    The numbers mean nothing; it checks that the code runs end to end."""
    began = time.time()
    cfg = smoke_config(cfg)
    root = run_root(cfg)
    if root.exists():
        shutil.rmtree(root)
    write_run_info(root, cfg, p)
    steps = []

    def done(what):
        steps.append(what)
        print(f"--- smoke: {what} ({time.time() - began:.0f} s)", flush=True)

    train_job(p, 2, 0, settings_for(cfg, 2, 0), root)
    done("trained and saved N = 2")
    train_job(p, 3, 0, settings_for(cfg, 3, 0), root, stop_after=1)
    done("N = 3 interrupted on purpose after round 1")
    train_job(p, 3, 0, settings_for(cfg, 3, 0), root)
    done("N = 3 resumed from its saved round and finished")
    folder = RunFolder(root, 3, 0)
    solution = load_solution(folder.final_path)
    x, y = np.linspace(0.0, 2.2, 5), np.array([1.0, 1.2])
    assert solution.position(0.5, x, y).shape == (5,) and solution.value(0.5, x, y).shape == (5,)
    done("loaded final.pt and read value / position")
    one_round = replace(settings_for(cfg, 3, 1), max_rounds=1, min_rounds=1)
    train_job(p, 3, 1, one_round, root, init_path=RunFolder(root, 2, 0).final_path)
    done("N = 3, seed 1 started from the N = 2 network (--init-from), one round")
    result = subprocess.run(
        [sys.executable, "scripts/analyze_nplayer.py", "--config", str(root / "config.yaml")],
        check=False,
    )
    assert result.returncode == 0, "the analysis script failed"
    done("analysis: Monte Carlo, references, table and figures")
    print(f"Smoke test passed in {time.time() - began:.0f} s. Covered: " + "; ".join(steps))


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/nplayer.yaml")
    parser.add_argument("--N", type=int, help="only this N")
    parser.add_argument("--seed", type=int, help="only this seed")
    parser.add_argument("--check", choices=["c1", "c2"], help="run an exact check instead")
    parser.add_argument("--init-from", help="a saved .pt to start from (any N, any code version)")
    parser.add_argument("--force", action="store_true", help="redo even what is finished")
    parser.add_argument("--plan", action="store_true", help="print the plan and exit")
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    parser.add_argument(
        "--time-steps", type=int, metavar="STEPS", help="time this many Adam steps per N"
    )
    parser.add_argument("--fresh-best-response", action="store_true", help="optional check C6")
    args = parser.parse_args()

    cfg, p = load_config(args.config)
    root = run_root(cfg)
    if args.plan:
        plan(cfg, root, jobs(cfg, args))
    elif args.smoke:
        smoke(cfg, p, args)
    elif args.check == "c2":
        check_c2(cfg, p, args)
    elif args.check == "c1":
        check_c1(cfg, p, args)
    elif args.time_steps:
        time_steps(cfg, p, args, root)
    elif args.fresh_best_response:
        fresh_best_response(cfg, p, args, root)
    else:
        train_all(cfg, p, args, root)


if __name__ == "__main__":
    main()
