# Mean-field portfolio game with a shortfall penalty

AMS 516 course project, Problem 21. A large population of investors each choose how much to put in a
stock. Each investor maximises

    E[ -exp(-γ X_T) - (λ/2) ((m̄_T - X_T)⁺)² ],

exponential utility of their own wealth minus a penalty for ending below the population average m̄_T.
The equilibrium couples a backward HJB equation (each investor's best strategy) with a forward
Fokker–Planck equation (how the wealth distribution evolves), which we solve with physics-informed
neural networks (PINNs) and validate against closed forms, finite differences and Monte Carlo.

The formulation, and why it differs from the original statement, is in
[`docs/problem/AMS516_Project21_proposal.pdf`](docs/problem/AMS516_Project21_proposal.pdf).

## Repository layout

```
configs/      Parameter sets (YAML). Every reported result should come from one of these.
src/mfg/      The package: parameters, utility, benchmarks, solvers.
tests/        Pytest checks against known answers. Run on every pull request.
scripts/      Entry points: train / solve / make figures.
notebooks/    Exploration only. Reusable code moves into src/mfg.
docs/         Problem statement, proposal, explanatory notes.
report/       Final report and the figures it uses.
results/      Experiment outputs (git-ignored; regenerate from configs).
```

## Setup

```bash
conda env create -f environment.yml
conda activate ams516-mfg
pip install -e ".[dev]"
pre-commit install
pytest
```

The CPU build of PyTorch is enough for these one-dimensional problems. To use an NVIDIA GPU, install
the matching CUDA build of PyTorch from https://pytorch.org/get-started/locally/ instead.

## Roadmap

Each item is a GitHub issue; tick it here when the issue is closed.

- [x] Parameters, utility and λ = 0 closed-form benchmarks, with tests
- [ ] Finite-difference solver for the HJB (fixed m̄_T)
- [ ] Finite-difference solver for the Fokker–Planck equation (given π*)
- [ ] Equilibrium loop on m̄_T with the FD solvers (reference solution for λ > 0)
- [ ] Monte Carlo simulation of N investors under a given π*
- [ ] PINN for V (validate against λ = 0 and FD)
- [ ] PINN for m (validate against FD)
- [ ] PINN equilibrium: alternating and joint training
- [ ] Sweep over the interaction strength; figures of π*(t, x), m̄_T and the wealth spread
- [ ] Report and presentation
