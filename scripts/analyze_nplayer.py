"""Monte Carlo, mean-field references, table and figures of the finite-N game, from the networks
saved by scripts/run_nplayer.py, into results/nplayer/<run_name>/analysis/.

Usage, from the repo folder with PYTHONPATH=src:
    python scripts/analyze_nplayer.py                               the run of configs/nplayer.yaml
    python scripts/analyze_nplayer.py --config results/nplayer/smoke/config.yaml
    python scripts/analyze_nplayer.py --resimulate       redo the Monte Carlo (else it is reused)
    python scripts/analyze_nplayer.py --extra-checks     C5 and "what ignoring the league costs"

What it writes in analysis/:
    mc_N05_seed0.npz        final (and starting) wealths of the games, their seed and source network
    references.json         1-D finite-difference mean-field numbers at lam_N (every N) and at lam
    table.csv, table.json   one row per N, then the mean-field rows
    fig_*.png               the figures: the network solid, finite differences / mean-field dashed
    summary.json            everything numeric, for notebooks/nplayer_example.ipynb
Rerunning only redraws: the games are kept with their seed and network, unless --resimulate.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import norm  # noqa: E402

from mfg.benchmarks import (  # noqa: E402
    c2_position,
    c2_value,
    lambda_n,
    merton_pi,
    population_mean,
    population_var,
)
from mfg.fd.equilibrium import solve_equilibrium  # noqa: E402
from mfg.montecarlo import group_statistics, mean_field_reference, simulate_games  # noqa: E402
from mfg.pinn.nets import pick_device  # noqa: E402
from mfg.pinn.nplayer import check_states  # noqa: E402
from mfg.pinn.nplayer_io import RunFolder, git_commit, load_solution  # noqa: E402
from mfg.utility import utility  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from run_nplayer import RESULTS, load_config, run_root, settings_for  # noqa: E402

# Plot style, as in the notebooks: the network in blues (light = small N, dark = large N), the
# finite-difference / mean-field references dashed orange, muted ink for text and guides.
INK, MUTED, BAND = "#0b0b0b", "#898781", "#f0efec"
NETWORK, REFERENCE = "#2a78d6", "#eb6834"
N_RAMP = ["#7fb0ea", "#5a97e3", "#3480dd", "#1c5cab", "#0d366b"]
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7",
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "axes.titlesize": 11,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "figure.dpi": 110,
    }
)


def n_colour(N, all_n):
    """One blue per N, lighter for smaller N (a sequential ramp, in the order of the config)."""
    order = sorted(all_n)
    return N_RAMP[min(order.index(N) * len(N_RAMP) // max(len(order), 1), len(N_RAMP) - 1)]


# ---------------------------------------------------------------- what is trained


def trained_jobs(cfg, root):
    """(N, seed, RunFolder, status) of every job with a saved network, in the config's order."""
    pairs = [(N, seed) for N in cfg["N"] for seed in cfg["seeds"]]
    for N, extra in (cfg.get("extra_seeds") or {}).items():
        pairs += [(int(N), seed) for seed in extra]
    found = []
    for N, seed in pairs:
        folder = RunFolder(root, N, seed)
        if folder.final_path.exists():
            found.append((N, seed, folder, folder.status() or {}))
    return found


# ---------------------------------------------------------------- Monte Carlo and references


def simulate(cfg, p, N, seed, folder, out, resimulate, device):
    """The games of one (N, seed), from the saved file if it exists."""
    path = out / f"mc_N{N:02d}_seed{seed}.npz"
    mc = cfg["montecarlo"]
    if path.exists() and not resimulate:
        return dict(np.load(path, allow_pickle=False))
    solution = load_solution(folder.final_path, device)
    games = simulate_games(
        solution.position,
        p,
        N,
        mc["investors_per_N"] // N,
        mc["n_steps"],
        mc["seed"],
        clip=tuple(mc["clip"]),
    )
    data = {
        "X_T": games.X_T,
        "X_0": games.X_0,
        "seed": mc["seed"],
        "n_steps": mc["n_steps"],
        "n_clipped": games.n_clipped,
        "source": str(folder.final_path),
        "git_commit": git_commit(),
    }
    np.savez(path, **data)
    return data


def references(cfg, p, out):
    """Mean-field numbers from the 1-D FD equilibrium: at lam_N for every N (the "my wealth moves
    the mean" effect alone) and at lam itself (N = infinity). Cached in references.json."""
    path = out / "references.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    band = cfg["montecarlo"]["band"]
    nx, nt = cfg["analysis"]["fd_nx"], cfg["analysis"]["fd_nt"]
    refs = {"lam": mean_field_reference(p, band, nx, nt), "lam_N": {}}
    for N in cfg["N"]:
        refs["lam_N"][str(N)] = mean_field_reference(p.with_(lam=lambda_n(p.lam, N)), band, nx, nt)
    path.write_text(json.dumps(refs, indent=1), encoding="utf-8")
    return refs


def extra_checks(cfg, p, N, folder, games, device, mf_rule):
    """C5: the mean utility of the games against the mean value V(0, .) on the same starting
    wealths (3 standard errors plus 0.5%). And what ignoring the league costs: the utility of
    investor 0 playing Merton, or the mean-field rule, while the others play the equilibrium."""
    mc = cfg["montecarlo"]
    solution = load_solution(folder.final_path, device)
    X_T, X_0 = games["X_T"], games["X_0"]
    utilities = utility(X_T[:, 0], X_T.mean(axis=1), p)
    V0 = solution.value(0.0, X_0[:, 0], X_0[:, 1:])
    stderr = utilities.std() / np.sqrt(len(utilities))
    out = {
        "c5_mean_utility": float(utilities.mean()),
        "c5_mean_value": float(V0.mean()),
        "c5_stderr": float(stderr),
        "c5_passed": bool(abs(utilities.mean() - V0.mean()) < 3 * stderr + 0.005 * abs(V0.mean())),
    }
    deviants = {
        "merton": lambda t, x, y: merton_pi(t, p) + 0 * x,
        "mean_field": lambda t, x, y: mf_rule(t, x),
    }
    for name, rule in deviants.items():
        other = simulate_games(
            solution.position,
            p,
            N,
            len(X_T),
            mc["n_steps"],
            mc["seed"],
            deviant=rule,
            clip=tuple(mc["clip"]),
        ).X_T
        gain = utilities.mean() - utility(other[:, 0], other.mean(axis=1), p).mean()
        out[f"cost_of_playing_{name}"] = float(gain)  # utility lost by investor 0 (same draws)
    return out


# ---------------------------------------------------------------- the table


def build_table(cfg, p, jobs, stats, refs):
    """One row per N (seeds pooled: seed 0's value, and the range over seeds if there are more),
    then the mean-field rows."""
    rows = []
    for N in cfg["N"]:
        seeds = [(seed, folder, status) for n, seed, folder, status in jobs if n == N]
        if not seeds:
            continue
        values = {
            key: [stats[(N, seed)][key] for seed, _, _ in seeds]
            for key in ("mean", "std", "crowding_index")
        }
        seed0, folder0, status0 = seeds[0]
        history = folder0.history()
        rows.append(
            {
                "N": N,
                "lam_N": lambda_n(p.lam, N),
                "seeds": len(seeds),
                "mean": values["mean"][0],
                "mean_stderr": stats[(N, seed0)]["mean_stderr"],
                "std": values["std"][0],
                "crowding_index": values["crowding_index"][0],
                "range_mean": max(values["mean"]) - min(values["mean"]),
                "range_std": max(values["std"]) - min(values["std"]),
                "range_crowding": max(values["crowding_index"]) - min(values["crowding_index"]),
                "rounds": status0.get("rounds_done"),
                "last_delta": (status0.get("deltas") or [None])[-1],
                "accepted": status0.get("status") == "accepted",
                "minutes": sum(h["seconds"] for h in history) / 60 if history else None,
                "mc_clipped": int(stats[(N, seed0)]["n_clipped"]),
            }
        )
    for N in cfg["N"]:
        ref = refs["lam_N"][str(N)]
        rows.append(mean_field_row(f"mean-field, lam_{N} = {ref['lam']:.3f}", ref))
    rows.append(mean_field_row(f"mean-field, lam = {p.lam:g}", refs["lam"]))
    return rows


def mean_field_row(name, ref):
    """A table row of a mean-field reference, with the columns the network rows have."""
    return {
        "N": name,
        "mean": ref["mbar"],
        "std": ref["std"],
        "crowding_index": ref["crowding_index"],
    }


def write_table(rows, out):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(out / "table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out / "table.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    print(
        f"{'N':>26} {'mean':>8} {'std':>7} {'crowding':>9} {'rounds':>6} "
        f"{'last delta':>10} {'accepted':>8}"
    )
    for row in rows:
        delta = row.get("last_delta")
        print(
            f"{str(row['N']):>26} {row['mean']:8.4f} {row['std']:7.4f} "
            f"{row['crowding_index']:9.3f} "
            f"{str(row.get('rounds', '')):>6} {'' if delta is None else f'{delta:.4f}':>10} "
            f"{str(row.get('accepted', '')):>8}"
        )


# ---------------------------------------------------------------- the figures


def quantile_rivals(N, t, p):
    """Rivals at the quantiles of the lam = 0 population at time t:
    y_j = M(t) + S(t) Phi^-1((j - 1/2) / (N - 1)). A realistic width; all rivals at one point
    (width zero) was never trained."""
    j = np.arange(1, N)
    return population_mean(t, p) + np.sqrt(population_var(t, p)) * norm.ppf((j - 0.5) / (N - 1))


def fig_rule_by_n(solutions, p, mf, out):
    """pi against my wealth at t = 0, 0.5 and 0.9, one curve per N (rivals at the population
    quantiles) and the mean-field rule dashed."""
    x = np.linspace(0.0, 2.2, 221)
    times = [0.0, 0.5, 0.9]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, t in zip(axes, times, strict=True):
        for N, sol in solutions.items():
            rivals = quantile_rivals(N, t, p)
            ax.plot(x, sol.position(t, x, rivals), color=n_colour(N, solutions), label=f"N = {N}")
        ax.plot(x, mf.hjb.position(t, x), color=REFERENCE, ls="--", label="mean-field (FD)")
        ax.axvline(population_mean(t, p), color=MUTED, lw=0.8)
        ax.set(title=f"t = {t}", xlabel="my wealth x")
    axes[0].set_ylabel("position pi (dollars in the stock)")
    axes[0].legend(loc="lower right", fontsize=9)
    title = "My rule by N, rivals at the population quantiles (grey line: their mean)"
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig2_rule_by_n.png")
    plt.close(fig)


def fig_ranking(solutions, p, out):
    """N = 10 (or the largest N trained): pi against my wealth at t = 0.5 with the rivals' mean
    fixed and the group more or less spread (width x0.5, x1, x1.5)."""
    N = 10 if 10 in solutions else max(solutions)
    sol, t = solutions[N], 0.5
    x = np.linspace(0.0, 2.2, 221)
    fig, ax = plt.subplots(figsize=(7, 3.9))
    for colour, factor in zip(["#9cc3f0", "#3480dd", "#0d366b"], [0.5, 1.0, 1.5], strict=True):
        rivals = population_mean(t, p) + factor * (quantile_rivals(N, t, p) - population_mean(t, p))
        ax.plot(x, sol.position(t, x, rivals), color=colour, label=f"group width x{factor}")
    ax.axvline(population_mean(t, p), color=MUTED, lw=0.8)
    title = f"N = {N}, t = {t}: the rivals' mean fixed, their spread changed"
    ax.set(title=title, xlabel="my wealth x", ylabel="position pi")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig3_ranking.png")
    plt.close(fig)


def fig_group_by_n(rows, refs, p, out):
    """Final mean, spread and crowding index against 1/N: the network's points (error bars: the
    range over seeds), the mean-field curve at lam_N dashed, and lam itself at 1/N = 0."""
    net_rows = [r for r in rows if isinstance(r["N"], int)]
    inv_n = [1 / r["N"] for r in net_rows]
    all_n = sorted(int(k) for k in refs["lam_N"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, key, ref_key, rng_key, title in zip(
        axes,
        ["mean", "std", "crowding_index"],
        ["mbar", "std", "crowding_index"],
        ["range_mean", "range_std", "range_crowding"],
        ["final mean of the group", "spread of final wealth", "crowding index (1 = no penalty)"],
        strict=True,
    ):
        ref_x = [1 / N for N in all_n] + [0.0]
        ref_y = [refs["lam_N"][str(N)][ref_key] for N in all_n] + [refs["lam"][ref_key]]
        ax.plot(
            ref_x,
            ref_y,
            color=REFERENCE,
            ls="--",
            marker="o",
            ms=4,
            label="mean-field at lam_N (FD), lam at 1/N = 0",
        )
        ax.errorbar(
            inv_n,
            [r[key] for r in net_rows],
            yerr=[r[rng_key] for r in net_rows],
            color=NETWORK,
            marker="o",
            ms=6,
            capsize=3,
            label="network (N-player game)",
        )
        for r in net_rows:
            ax.annotate(
                f"N = {r['N']}",
                (1 / r["N"], r[key]),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=8,
                color=MUTED,
            )
        ax.set(title=title, xlabel="1 / N")
    axes[0].legend(loc="best", fontsize=8)
    title = "The group by N: the network against the mean-field with the same lam_N"
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig4_group_by_n.png")
    plt.close(fig)


def fig_cost(rows, out):
    """Finite differences would need 800^N grid points per instant; the network's measured
    training time grows slowly with N. Two panels, one quantity each."""
    net_rows = [r for r in rows if isinstance(r["N"], int) and r.get("minutes")]
    if not net_rows:
        return
    n_values = [r["N"] for r in net_rows]
    fig, (ax_fd, ax_net) = plt.subplots(1, 2, figsize=(11, 3.9))
    ax_fd.semilogy(n_values, [800.0**N for N in n_values], color=REFERENCE, ls="--", marker="o")
    ax_fd.axhline(1e9, color=MUTED, lw=0.8)
    note = "about the memory of a laptop (8 GB of doubles)"
    ax_fd.text(n_values[0], 0.7e9, note, color=MUTED, fontsize=8, va="top")
    title = "finite differences: grid points per instant (800 per wealth)"
    ax_fd.set(title=title, xlabel="N", ylabel="points")
    ax_net.plot(n_values, [r["minutes"] for r in net_rows], color=NETWORK, marker="o")
    ax_net.set(title="network: measured training time, all rounds", xlabel="N", ylabel="minutes")
    fig.tight_layout()
    fig.savefig(out / "fig5_cost.png")
    plt.close(fig)


def fig_c2(cfg, p, out, device):
    """Check C2, if it was run: the errors of pi and V on its check states against t."""
    conf = cfg["checks"]["c2"]
    N, eta = int(conf["N"]), float(conf["eta"])
    folder = RunFolder(RESULTS / "checks" / "c2", N, cfg["seeds"][0])
    if not folder.final_path.exists():
        return None
    solution = load_solution(folder.final_path, device)
    s = settings_for(cfg, N, cfg["seeds"][0])
    states = check_states(p, N, s.n_check, s.check_seed, s.check_t_max)
    t, x, y = states["t"], states["x"], states["y"]
    pi_error = np.abs(solution.position(t, x, y) - c2_position(t, N, eta, p))
    V_exact = c2_value(t, x, y, N, eta, p)
    V_error = np.abs(solution.value(t, x, y) - V_exact) / np.abs(V_exact)
    fig, (ax_pi, ax_v) = plt.subplots(1, 2, figsize=(11, 3.9))
    ax_pi.plot(t, pi_error, ".", color=NETWORK, ms=2, alpha=0.3)
    for level, text in [(0.02, "mean target 0.02"), (0.07, "max target 0.07")]:
        ax_pi.axhline(level, color=MUTED, lw=0.8)
        ax_pi.text(0.0, level, text, color=MUTED, fontsize=8, va="bottom")
    title = f"C2, N = {N}: |pi - exact| on the check states"
    ax_pi.set(title=title, xlabel="t", ylabel="error in pi")
    ax_v.plot(t, 100 * V_error, ".", color=NETWORK, ms=2, alpha=0.3)
    ax_v.axhline(0.5, color=MUTED, lw=0.8)
    ax_v.text(0.0, 0.5, "target 0.5%", color=MUTED, fontsize=8, va="bottom")
    ax_v.set(title="C2: relative error of V (%)", xlabel="t", ylabel="% error")
    fig.tight_layout()
    fig.savefig(out / "fig1_c2_check.png")
    plt.close(fig)
    return {
        "pi_mean_error": float(pi_error.mean()),
        "pi_max_error": float(pi_error.max()),
        "V_max_relative_error": float(V_error.max()),
    }


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/nplayer.yaml")
    parser.add_argument("--resimulate", action="store_true", help="redo the Monte Carlo")
    parser.add_argument(
        "--extra-checks", action="store_true", help="C5 and the cost of ignoring the league"
    )
    args = parser.parse_args()

    cfg, p = load_config(args.config)
    root = run_root(cfg)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    jobs = trained_jobs(cfg, root)
    if not jobs:
        sys.exit(f"nothing trained yet in {root}")
    print(f"Analysing {len(jobs)} trained networks in {root}")

    refs = references(cfg, p, out)
    # The mean-field equilibrium at lam itself: its rule is the dashed reference of the figures.
    mf = solve_equilibrium(p, cfg["analysis"]["fd_nx"], cfg["analysis"]["fd_nt"])
    stats, extras = {}, {}
    for N, seed, folder, _status in jobs:
        games = simulate(cfg, p, N, seed, folder, out, args.resimulate, device)
        st = group_statistics(games["X_T"], p, cfg["montecarlo"]["band"])
        stats[(N, seed)] = {**st, "n_clipped": int(games["n_clipped"])}
        print(
            f"N = {N:2d}, seed {seed}: mean {st['mean']:.4f}, std {st['std']:.4f}, "
            f"crowding {st['crowding_index']:.3f}, MC clips {games['n_clipped']}"
        )
        if args.extra_checks:
            extra = extra_checks(cfg, p, N, folder, games, device, mf.hjb.position)
            extras[f"N{N}_seed{seed}"] = extra
            print("   ", json.dumps(extra))

    rows = build_table(cfg, p, jobs, stats, refs)
    write_table(rows, out)

    solutions = {
        N: load_solution(folder.final_path, device)
        for N, seed, folder, _ in jobs
        if seed == cfg["seeds"][0]
    }
    fig_rule_by_n(solutions, p, mf, out)
    fig_ranking(solutions, p, out)
    fig_group_by_n(rows, refs, p, out)
    fig_cost(rows, out)
    c2 = fig_c2(cfg, p, out, device)

    summary = {
        "run": cfg["run_name"],
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "table": rows,
        "references": refs,
        "statistics": {f"N{N}_seed{seed}": stats[(N, seed)] for N, seed, _, _ in jobs},
        "c2": c2,
        "extra_checks": extras,
    }
    text = json.dumps(summary, indent=1, default=float)
    (out / "summary.json").write_text(text, encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
