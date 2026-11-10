"""Saving and reloading the finite-N networks, so that hours of training are never repeated.

A run lives in results/nplayer/<run_name>/N05_seed0/ (RunFolder): one .pt per round, final.pt,
history.json (per round: the change Delta, the loss parts, the L-BFGS guard's winner, the clips,
the seconds and the settings), pi_check.npz (the rule on the check states after every round,
which does not depend on the code) and status.json. A relaunch continues from the last round
saved (RunFolder.start_point), and --init-from starts a run from any saved network, of another N
or another version of the code (init_from: the tensors that fit are loaded, the rest reported).

Each .pt holds only plain types and tensors, so torch.load(path, weights_only=True) works:
format_version, state_dict, net_class, net_kwargs, params, N, lam, lam_N, round, delta,
settings, git_commit and created.
"""

import json
import shutil
import subprocess
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from mfg.benchmarks import lambda_n
from mfg.params import Params
from mfg.pinn.nets import SymmetricNet
from mfg.pinn.nplayer import NPlayerSettings, NPlayerSolution, RoundsStart

FORMAT_VERSION = 1


def git_commit():
    """The short hash of the current commit, or "unknown" outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def checkpoint(net, p: Params, N, round_, delta, settings: NPlayerSettings):
    """The dictionary saved in a .pt file (plain types and tensors only)."""
    return {
        "format_version": FORMAT_VERSION,
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "net_class": type(net).__name__,
        "net_kwargs": dict(net.kwargs),
        "params": asdict(p),
        "N": int(N),
        "lam": float(p.lam),
        "lam_N": float(lambda_n(p.lam, N)),
        "round": int(round_),
        "delta": float(delta),
        "settings": asdict(settings),
        "git_commit": git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def save_checkpoint(path, net, p: Params, N, round_, delta, settings: NPlayerSettings):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint(net, p, N, round_, delta, settings), path)


def load_checkpoint(path, device=None):
    """The saved dictionary, with the network rebuilt under "net" (strict: the same architecture).

    Only the network's own kwargs matter for the scales and sizes; Params come from the file.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    p = Params(**saved["params"])
    net = SymmetricNet(p, **saved["net_kwargs"])
    net.load_state_dict(saved["state_dict"])
    saved["net"] = net.to(device or "cpu")
    saved["p"] = p
    return saved


def load_solution(path, device=None) -> NPlayerSolution:
    """A saved network ready for value(t, x, y) and position(t, x, y), on the CPU or a GPU."""
    saved = load_checkpoint(path, device)
    return NPlayerSolution(saved["net"], saved["p"], saved["N"])


def load_matching_weights(net, state_dict):
    """Copy into `net` the saved tensors whose name and shape fit; warn about the rest.

    For --init-from across versions of the code or sizes of the network. Returns the names
    skipped (missing in the file or of another shape) and the names of `net` left untouched.
    """
    own = net.state_dict()
    fitting, skipped = {}, []
    for name, tensor in state_dict.items():
        if name in own and own[name].shape == tensor.shape:
            fitting[name] = tensor
        else:
            skipped.append(name)
    untouched = [name for name in own if name not in fitting]
    net.load_state_dict(fitting, strict=False)
    if skipped or untouched:
        warnings.warn(
            f"{len(skipped)} saved tensors did not fit the network ({skipped}); "
            f"{len(untouched)} of its tensors keep their start values ({untouched})",
            UserWarning,
            stacklevel=2,
        )
    return skipped, untouched


def init_from(path, p: Params, settings: NPlayerSettings, device=None) -> SymmetricNet:
    """A network of the current settings started from the weights saved at `path`.

    The same weights serve any N (the rivals' summaries are averaged), so N = 20 can start from
    the network of N = 10. If the architecture changed, the tensors that fit are loaded and the
    rest reported, never silently.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    net = SymmetricNet(p, **asdict(settings.net))
    load_matching_weights(net, saved["state_dict"])
    return net.to(device or "cpu")


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class RunFolder:
    """The files of one (N, seed) of a run: results/nplayer/<run_name>/N05_seed0/."""

    def __init__(self, root, N, seed):
        self.N, self.seed = N, seed
        self.folder = Path(root) / f"N{N:02d}_seed{seed}"
        self.final_path = self.folder / "final.pt"
        self.history_path = self.folder / "history.json"
        self.pi_check_path = self.folder / "pi_check.npz"
        self.status_path = self.folder / "status.json"

    def round_path(self, k):
        return self.folder / f"round_{k:02d}.pt"

    def rounds_saved(self):
        return sorted(int(f.stem.split("_")[1]) for f in self.folder.glob("round_*.pt"))

    def status(self):
        return _read_json(self.status_path) if self.status_path.exists() else None

    def history(self):
        return _read_json(self.history_path) if self.history_path.exists() else []

    def write_status(self, status, **more):
        self.folder.mkdir(parents=True, exist_ok=True)
        _write_json(
            self.status_path,
            {
                "status": status,
                "N": self.N,
                "seed": self.seed,
                "rounds_done": len(self.rounds_saved()),
                "updated": datetime.now().isoformat(timespec="seconds"),
                **more,
            },
        )

    def clear(self):
        """Start again (--force)."""
        if self.folder.exists():
            shutil.rmtree(self.folder)

    def save_round(self, result, net, p: Params, N, settings: NPlayerSettings, states=None):
        """Called after every round (solve_nplayer_equilibrium's on_round): the network of the
        round and of final.pt, the history entry, the rule on the check states and the status."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.round_path(result.round)
        save_checkpoint(path, net, p, N, result.round, result.delta, settings)
        shutil.copyfile(path, self.final_path)

        training = result.training
        entry = {
            "round": result.round,
            "delta": result.delta,
            "seconds": training.seconds,
            "stage": asdict(result.stage),
            "guard_winner": training.guard_winner,
            "n_floor": training.n_floor,
            "n_cap": training.n_cap,
            "loss": {k: v[-1] for k, v in training.history.items() if k != "step"},
            "c3": result.c3,
            "loss_history": training.history,
        }
        history = [h for h in self.history() if h["round"] < result.round] + [entry]
        _write_json(self.history_path, history)

        if self.pi_check_path.exists():
            saved = dict(np.load(self.pi_check_path))
            pi = np.concatenate([saved["pi"][: result.round - 1], result.pi_check[None]])
            np.savez(self.pi_check_path, **{**saved, "pi": pi})
        else:
            extra = {} if states is None else {k: states[k] for k in ("t", "x", "y")}
            np.savez(self.pi_check_path, pi=result.pi_check[None], **extra)
        self.write_status("running")

    def finish(self, equilibrium):
        """Mark the run accepted or not, with the deltas of every round."""
        self.write_status(
            "accepted" if equilibrium.accepted else "not_accepted",
            deltas=[float(d) for d in equilibrium.deltas],
            seconds=sum(h["seconds"] for h in self.history()),
        )

    def start_point(self, device=None):
        """Where to resume: the network, the deltas and the rule of the last round saved, or None
        if nothing is saved yet."""
        rounds = self.rounds_saved()
        if not rounds:
            return None
        last = rounds[-1]
        saved = load_checkpoint(self.round_path(last), device)
        deltas = [h["delta"] for h in self.history() if h["round"] <= last]
        pi_before = np.load(self.pi_check_path)["pi"][last - 1]
        return RoundsStart(saved["net"], rounds_done=last, deltas=deltas, pi_before=pi_before)
