"""Model parameters, loadable from the YAML files in configs/."""

from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Params:
    r: float = 0.02
    mu: float = 0.08
    sigma: float = 0.20
    gamma: float = 1.0
    lam: float = 2.0
    T: float = 1.0
    m0_mean: float = 1.0
    m0_std: float = 0.2
    x_min: float = -1.0
    x_max: float = 3.0

    @property
    def kappa(self) -> float:
        """Sharpe ratio (mu - r) / sigma."""
        return (self.mu - self.r) / self.sigma

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Params":
        # Read as UTF-8, otherwise Windows uses its own encoding and fails on Greek letters.
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        flat = {k: v for section in raw.values() for k, v in section.items()}
        return cls(**flat)

    def with_(self, **changes) -> "Params":
        """Copy with some fields changed, e.g. p.with_(lam=0.0)."""
        return replace(self, **changes)
