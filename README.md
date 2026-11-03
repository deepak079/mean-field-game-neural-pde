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
src/mfg/      The package: parameters, utility, benchmarks; solvers in fd/ and pinn/.
tests/        Pytest checks against known answers. Run on every pull request.
scripts/      Entry points: train / solve / make figures.
notebooks/    Exploration only. Reusable code moves into src/mfg.
docs/         Problem statement, proposal and task summaries.
report/       Final report and the figures it uses.
results/      Experiment outputs (git-ignored; regenerate from configs).
```

## Setup

```bash
conda env create -f environment.yml
conda activate ams516-mfg
pip install -e ".[dev]"
pytest
```

Optional tidy-up before opening a pull request (formats code, strips notebook outputs):

```bash
pre-commit run --all-files
```

`environment.yml` installs the CPU build of PyTorch (`pytorch-cpu` from conda-forge), which is enough
for these one-dimensional problems. To use an NVIDIA GPU, replace it with `pytorch-gpu`. Install
PyTorch with conda rather than pip in this environment: pip's build can load an older system C++
library and break conda's numpy.

## Roadmap

DS works on the finite-difference side and AI on the PINN side. Tick a task in the pull request
that finishes it. Unassigned tasks: agree in chat who takes them, then add your initials.

- [x] DS: Parameters, utility and λ = 0 closed-form benchmarks, with tests
- [x] DS: Notes on the HJB and the Fokker–Planck equation (kept outside the repo)
- [x] DS: Lower-boundary value for the HJB in `benchmarks.py` (shared by both solvers)
- [x] DS: Finite-difference solver for the HJB (fixed m̄_T)
- [x] DS: Finite-difference solver for the Fokker–Planck equation (given π*)
- [x] DS: Equilibrium on m̄_T with the FD solvers (secant on Φ(m̄) = m̄; reference for λ > 0)
- [x] AI: PINN for V at λ = 0 (validate against `merton_value`)
- [x] AI: PINN for V at λ = 2, m̄ = 1.1 fixed (validate against the FD solver)
- [x] DS (taken over from AI): PINN for m (validate against the Gaussian at λ = 0, then FD)
- [x] DS (taken over from AI): PINN equilibrium, alternating training with the secant outer loop (m̄ within 1.5e−4 of FD; joint training left out)
- [ ] Monte Carlo simulation of N investors under a given π*
- [ ] Comparison script: FD vs PINN for V, π* and m
- [ ] Sweep over the interaction strength; figures of π*(t, x), m̄_T and the wealth spread
- [ ] Report and presentation
