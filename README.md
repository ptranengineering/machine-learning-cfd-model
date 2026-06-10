# Aero-ML Project

Neural-network and ensemble surrogates for aerodynamic design: automate SU2 CFD, build datasets, train comparably evaluated surrogate models, optimize designs, and re-validate optima with fresh CFD.

The **SU2 design-space pipeline** is the primary research path. A **legacy Solidworks / PyTorch NN** track remains for commercial CFD exports.

---

## Table of Contents

1. [What This Program Does Now](#what-this-program-does-now)
2. [End-to-End Workflow](#end-to-end-workflow)
3. [Development History (Iterative Design)](#development-history-iterative-design)
4. [Research Focus Areas](#research-focus-areas)
5. [Repository Layout](#repository-layout)
6. [Setup](#setup)
7. [Pipeline A: SU2 Design-Space Surrogate (Primary)](#pipeline-a-su2-design-space-surrogate-primary)
8. [Evaluation and Validation](#evaluation-and-validation)
9. [Pipeline B: SU2 AoA Sweep (Fixed Geometry)](#pipeline-b-su2-aoa-sweep-fixed-geometry)
10. [Pipeline C: Legacy Solidworks NN (Cd / Cl / Cm)](#pipeline-c-legacy-solidworks-nn-cd--cl--cm)
11. [Module Reference](#module-reference)
12. [Models, Features, and Metrics](#models-features-and-metrics)
13. [Design Optimization](#design-optimization)
14. [Data Formats and Artifacts](#data-formats-and-artifacts)
15. [Configuration](#configuration)
16. [Dependencies](#dependencies)
17. [Troubleshooting](#troubleshooting)
18. [Literature Positioning](#literature-positioning)
19. [References](#references)

---

## What This Program Does Now

| Capability | Description |
|------------|-------------|
| **Design-space sampling** | LHS or random 6D sampling (geometry thickness/camber/position, AoA, Mach, Re) |
| **CFD execution (default)** | Automated **SU2 RANS + Spalart–Allmaras** with parametric NACA mesh deformation, Reynolds applied per case, CFL backoff retries, residual checks, sweep manifests |
| **CFD execution (legacy)** | Euler inviscid via `--solver euler` |
| **Scalable sweeps** | Resume (`--skip-completed`), append (`--append-raw-output`), batch script for 500+ cases |
| **Dataset building** | Two-tier quality gate (`quality_pass` / `strict_pass`), per-case quality reports, dedupe across runs |
| **Surrogate training** | Linear, Random Forest, Extra Trees, **Gaussian Process**; optional engineered features; CV model selection; readiness gate |
| **Unified metrics** | MAE, RMSE, R², MAPE / relative % error; comparison tables (CSV + JSON) |
| **Optimization** | Bayesian optimization (acquisition GP on surrogate scores) with CL/CD constraints |
| **CFD re-validation** | Re-run SU2 on optimized designs; compare predicted vs. fresh CFD; training-hull check |
| **Research reporting** | Model comparison summary, research readiness assessment (JSON + Markdown) |
| **User interfaces** | Web (FastAPI), desktop popup (Tkinter), interactive CLI |
| **Legacy track** | PyTorch NN on Solidworks exports; AoA fixed-geometry SU2 sweep |

### Measured status (local development run, 44 accepted RANS cases)

These numbers come from the current local dataset and are **not** guaranteed on a fresh install — regenerate locally to reproduce.

| Item | Value |
|------|-------|
| CFD success rate | ~80% (44/54 raw rows accepted after quality gate) |
| Strict convergence | 6/44 cases (`strict_pass`) |
| Best CV model | Extra Trees — CL R² ≈ 0.62, CD R² ≈ 0.77 |
| Readiness gate | **FAIL** (targets: CV R² ≥ 0.95, MAE ≤ 5% of range) |
| Optimization validation | Surrogate CL/CD 1.56/0.038 vs. CFD 2.50/−0.06 at extrapolated optimum (non-physical CFD) |

---

## End-to-End Workflow

```
generate_design_space.py     →  design_space.csv
        ↓
run_design_sweep.py          →  aero_design_raw.csv  (+ case dirs, sweep_manifest.json)
        ↓
build_dataset.py             →  aero_design_dataset.csv  +  aero_design_quality_report.csv
        ↓
train_design_baseline.py     →  design_rf_model.joblib  +  design_baseline_metrics.json
        ↓
generate_model_comparison.py →  model_comparison_summary.csv / .json
        ↓
optimize_design.py           →  design_optimization_result.json
        ↓
validate_optimization.py     →  optimization_validation.json / .csv
        ↓
generate_research_readiness_report.py  →  research_readiness_report.md / .json
```

All commands run from the repository root with `./.venv/bin/python` (see [Setup](#setup)).

---

## Development History (Iterative Design)

The framework was upgraded from an Euler proof-of-concept to a research-oriented RANS pipeline through incremental fixes discovered during real SU2 runs. This section documents **what broke, why, and how it was fixed**.

### Phase 1 — RANS migration (SU2 8.5)

| Problem | Cause | Fix |
|---------|-------|-----|
| `KIND_TURB_MODEL must be NONE if SOLVER= NAVIER_STOKES` | In SU2 8.5, `NAVIER_STOKES` is **laminar only**; turbulence requires `SOLVER= RANS` | Added `su2_cases/rans_NACA0012.cfg` with `SOLVER= RANS`, `KIND_TURB_MODEL= SA`, `MARKER_HEATFLUX` wall BCs, `REYNOLDS_NUMBER` |
| Reynolds ignored in design sweeps | Euler cfg had no `REYNOLDS_NUMBER` key | `run_design_sweep.py` always injects Reynolds; shared helpers in `scripts/su2_utils.py` |
| Default solver still Euler | Historical `inv_NACA0012.cfg` default | `--solver rans` is now the default; `--solver euler` retained for comparison |

### Phase 2 — Convergence and coefficient acceptance

| Problem | Cause | Fix |
|---------|-------|-----|
| `success=0` despite SU2 exit success | Residual thresholds (-6.0 / drop 4.0) too strict for transonic RANS on coarse mesh; valid CL/CD discarded on retry | RANS-specific defaults (`-4.0` / `2.0`); **marginal acceptance** returns `status=success` with `converged=false` when SU2 exits 0 and coefficients are physical |
| `ITER=1000` insufficient | Transonic cases need more iterations | Increased to `ITER=3000` in RANS cfg |
| No convergence metadata in design track | Design sweep lacked AoA-track features | Persist `history.csv`, `case_meta.json`, residual columns in raw CSV |

### Phase 3 — Dataset quality gate

| Problem | Cause | Fix |
|---------|-------|-----|
| `build_dataset.py` aborted when any row failed | Strict gate treated partial batch failure as total failure | **Two-tier gate**: `quality_pass` (success + physical CL/CD) vs. `strict_pass` (+ convergence residuals). Rejected rows excluded with warning; dataset still written |
| Single smoke-test case rejected | `converged=false` failed strict convergence check | Base `quality_pass` no longer requires `converged=true`; use `--require-strict-convergence` for high-confidence subsets |
| Legacy Euler CSV incompatible | New columns missing on old rows | Gate skips convergence checks when residual columns absent |

### Phase 4 — ML training and evaluation

| Problem | Cause | Fix |
|---------|-------|-----|
| Training blocked at N=1 | Hard minimum of 8 samples | `--smoke-test` mode for pipeline checks; normal training still requires ≥8 by default |
| No GP baseline | Only RF/LR/ET in design baselines | Added `gaussian_process` to `train_design_baseline.py` with shared `ml/metrics_utils.py` |
| No cross-model comparison report | Metrics scattered per script | `scripts/generate_model_comparison.py` → CSV + JSON tables |
| GP `ConvergenceWarning` spam | 25 CV folds × small N | Expected at N<100; RF/Extra Trees preferred until dataset scales |

### Phase 5 — Optimization validation (closed loop)

| Problem | Cause | Fix |
|---------|-------|-----|
| No surrogate→CFD loop | Optimizer returned surrogate-only predictions | Added `ml/validate_optimization.py` |
| Validation returned `CFD CL=None` | `run_case()` dropped coefficients when `physical_bounds` failed | `capture_last_coefficients=True` reports last parsed CL/CD even on failure |
| Optimizer extrapolated outside data | Default bounds wider than training hull | Validation reports `within_training_hull`; measured gap at N=44 (predicted L/D ≈ 41 vs. non-physical CFD) |

### Phase 6 — Documentation and positioning

- Added `docs/literature_positioning.md` (comparison vs. SMT, Dakota, generic ML workflows)
- Added `scripts/generate_research_readiness_report.py`
- Softened unsupported claims in `ml/README.md`; metrics drive readiness statements here

---

## Research Focus Areas

Open research directions given **current measured performance** (44-case RANS dataset, readiness FAIL).

### 1. Surrogate accuracy vs. CFD budget (data efficiency)

**Question:** How many converged RANS cases are needed to reach the readiness target?

- Readiness gate: mean CV R² ≥ 0.95 and MAE ≤ **5% of target range** per output (CL, CD) — not MAPE.
- At N=44: Extra Trees CV R² ≈ 0.62 (CL), 0.77 (CD); gate still **FAIL**.
- **Next steps:** Learning curves at N = 50, 100, 200, 500; `run_large_design_sweep.sh`; compare `quality_pass` vs. `strict_pass` training subsets.

### 2. Physics fidelity and model validity

**Question:** How trustworthy are surrogates trained on this RANS setup?

- Default: **RANS + Spalart–Allmaras**, Reynolds applied, transonic Mach band.
- Geometry: parametric NACA deformation on a shared coarse mesh (y+ not optimized).
- Closed-loop validation exists: `validate_optimization.py` demonstrated large surrogate–CFD gap at extrapolated optima.
- **Next steps:** Euler vs. RANS ablation; mesh refinement; training-hull constraints on optimizer.

### 3. Feature engineering and architecture selection

**Question:** Do nonlinear input transforms improve generalization in sparse design spaces?

- Design baselines support `--engineer-features` (sin/cos AoA, log Re, interaction terms).
- A separate `train_design_nn.py` exists but is **not wired into the optimizer**.
- Legacy NN has `plot_parameter_sensitivity()` but it is **not called** during training.
- **Research actions:** Ablation study (base 6D vs. 13D features); compare RF vs. design NN vs. physics-informed architectures; enable sensitivity plots for interpretability.

### 4. Optimization under surrogate uncertainty

**Question:** Can we optimize safely when the surrogate is wrong outside the training hull?

- Optimizer uses Bayesian optimization on **surrogate-predicted** CL/CD (acquisition GP fits scalar scores, not physics).
- Readiness gate is **advisory** — optimization runs when status is `FAIL`.
- **CFD re-validation** is implemented (`validate_optimization.py`); measured extrapolation failure at N=44.
- **Next steps:** Constrain search to training hull; enforce readiness before optimize; trust-region penalties.

### 5. Multi-objective and constraint handling

**Question:** How do CL/CD trade-offs behave across the design space?

- Objectives: `max_cl_cd`, `max_cl`, `min_cd` with box constraints (`min_cl`, `max_cd`).
- Moment coefficient **Cm is not predicted** in SU2 pipelines (only CL/CD).
- **Research actions:** Pareto fronts; constrained multi-objective BO; extend outputs to Cm and pitching stability metrics.

### 6. Convergence-aware learning (AoA sweep track)

**Question:** Can residual history features improve predictions when CFD quality varies?

- AoA dataset builder extracts `rms_rho_*`, `convergence_rate`, `has_history` from `history.csv`.
- Strict quality gate: positive CD, |CL| ≤ 3, history present, RMS drop thresholds.
- **Research actions:** Train `train_nn.py` with/without convergence features; study whether bad cases should be dropped vs. modeled.

### 7. Reproducibility and deployment

- Datasets and trained models are **gitignored**; every collaborator must regenerate locally.
- Web deploy requires mounting or generating `design_rf_model.joblib` on the host.
- **Research actions:** Document exact SU2 version, seed, and sweep manifests; publish dataset hashes; containerize full pipeline.

### 8. Evaluation targets (advisory readiness gate)

| Target | Metric | Where checked | Status at N=44 |
|--------|--------|---------------|----------------|
| Surrogate fidelity | CV R² ≥ 0.95, MAE ≤ 5% of range | `design_baseline_metrics.json` | **FAIL** |
| Generalization | Holdout + RepeatedKFold CV | `train_design_baseline.py` | CV R² ≈ 0.62–0.77 |
| Physical plausibility | CD > 0, \|CL\| ≤ 3 | `run_design_sweep.py`, `build_dataset.py` | ~80% CFD yield |
| Optimization validation | Predicted vs. fresh CFD | `optimization_validation.json` | Large gap at extrapolated optimum |

---

## Repository Layout

```
aero-ml-project/
├── docs/
│   └── literature_positioning.md    # vs. SMT, Dakota, generic ML workflows
├── ml/
│   ├── train_design_baseline.py   # LR / RF / ET / GP + readiness gate
│   ├── validate_optimization.py   # Surrogate vs. fresh CFD on optima
│   ├── metrics_utils.py           # Shared MAE/RMSE/R²/MAPE metrics
│   ├── optimize_design.py         # Bayesian optimization
│   ├── design_feature_utils.py    # Engineered input features
│   └── ...                        # Legacy NN, AoA trainers, UIs
├── scripts/
│   ├── su2_utils.py               # Shared convergence checks, coeff parsing
│   ├── generate_design_space.py
│   ├── run_design_sweep.py        # RANS/Euler design sweeps + manifest
│   ├── build_dataset.py           # Two-tier quality gate + reports
│   ├── run_large_design_sweep.sh  # 500+ case batch with resume
│   ├── generate_model_comparison.py
│   ├── generate_research_readiness_report.py
│   ├── generate_all_figures.py
│   ├── run_sweep.sh               # AoA sweep
│   └── full_pipeline.sh
├── su2_cases/
│   ├── rans_NACA0012.cfg          # Default: RANS + SA
│   ├── inv_NACA0012.cfg            # Legacy Euler
│   ├── mesh_NACA0012_inv.su2
│   ├── design_runs/               # Gitignored sweep outputs
│   └── validation_runs/           # Gitignored optimization re-runs
├── webapp/
├── datasets/                      # Gitignored
├── results/                       # Gitignored
└── requirements.txt
```

---

## Setup

### 1. Python environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r ml/requirements.txt # if using ml/ standalone
```

For the web UI, also install `pip install -r webapp/requirements.txt`.

### 2. SU2 (required for Pipelines A and B)

Install [SU2](https://su2code.github.io/) and ensure `SU2_CFD` is on your `PATH`:

```bash
which SU2_CFD
```

### 3. GPU (optional)

PyTorch training uses CUDA when available. Verify:

```python
import torch
print(torch.cuda.is_available())
```

### 4. Desktop popup UI (optional)

On Linux/WSL, install Tkinter: `sudo apt install python3-tk`

---

## Pipeline A: SU2 Design-Space Surrogate (Primary)

Trains on `datasets/processed/aero_design_dataset.csv` (rebuilt locally; not in Git).

### Step 1 — Sample the design space

```bash
./.venv/bin/python scripts/generate_design_space.py --n-samples 200 --seed 1
```

**Output:** `datasets/raw/design_space.csv`

Default bounds are conservative for SU2 convergence (e.g. AoA 0–6°, thickness 0.10–0.16). Override per dimension with `--geometry-thickness-range`, `--aoa-range`, etc.

### Step 2 — Run SU2 for each sample

```bash
./.venv/bin/python scripts/run_design_sweep.py --solver rans --retry-max 5
```

For 500+ cases with resume:

```bash
N_SAMPLES=500 ./scripts/run_large_design_sweep.sh
```

**Outputs:**
- `datasets/raw/aero_design_raw.csv`
- `su2_cases/design_runs/<run_id>/case_*/` (mesh, cfg, forces, logs)

Accumulate data across runs (resume skips already-successful `design_id`s):

```bash
./.venv/bin/python scripts/run_design_sweep.py --solver rans --retry-max 2 \
  --append-raw-output --skip-completed
```

### Step 3 — Build processed dataset

```bash
./.venv/bin/python scripts/build_dataset.py --dataset-type design \
  --design-raw datasets/raw/aero_design_raw.csv
```

Merges multiple raw files, dedupes on `(thickness, camber, pos, AoA, Mach, Re)` (last wins), applies convergence + physical quality gate.

**Outputs:**
- `datasets/processed/aero_design_dataset.csv`
- `datasets/processed/aero_design_quality_report.csv` (accepted vs. rejected)

### Step 4 — Train surrogate

```bash
./.venv/bin/python ml/train_design_baseline.py --engineer-features
```

Pipeline check with very few samples:

```bash
./.venv/bin/python ml/train_design_baseline.py --engineer-features --smoke-test
```

**Outputs:**
- `results/models/design_rf_model.joblib` (best CV model, despite filename)
- `results/design_baseline_metrics.json` (holdout + CV + readiness + `comparison_table`)

### Step 5 — Optimize

```bash
# Batch CLI with JSON bounds
./.venv/bin/python ml/optimize_design.py --bounds-json path/to/bounds.json --objective max_cl_cd

# Interactive terminal prompts
./.venv/bin/python ml/interactive_optimize_design.py

# Desktop popup (sliders)
./.venv/bin/python ml/popup_optimize_design.py

# Web UI
uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000
# → http://127.0.0.1:8000
```

See [webapp/README.md](webapp/README.md) for deployment (Render, Docker).

---

## Evaluation and Validation

### Model comparison table

```bash
./.venv/bin/python scripts/generate_model_comparison.py
```

**Outputs:** `results/model_comparison_summary.csv`, `results/model_comparison_summary.json`

### Optimization CFD re-validation

```bash
./.venv/bin/python ml/optimize_design.py
./.venv/bin/python ml/validate_optimization.py
```

**Outputs:** `results/optimization_validation.json`, `results/optimization_validation.csv`

The validation report includes predicted vs. CFD CL/CD, percent errors, `failure_reason`, and `within_training_hull` (whether the optimum lies inside the training dataset bounds per dimension).

### Research readiness report

```bash
./.venv/bin/python scripts/generate_research_readiness_report.py
```

**Outputs:** `results/research_readiness_report.md`, `results/research_readiness_report.json`

### Dataset quality options

```bash
# Default: accept marginal converged cases (quality_pass)
./.venv/bin/python scripts/build_dataset.py --dataset-type design

# High-confidence subset only (strict_pass: converged + residual thresholds)
./.venv/bin/python scripts/build_dataset.py --dataset-type design --require-strict-convergence

# Abort if any row fails gate (old strict behavior)
./.venv/bin/python scripts/build_dataset.py --dataset-type design --fail-on-rejects
```

---

## Pipeline B: SU2 AoA Sweep (Fixed Geometry)

Studies lift/drag vs. angle of attack on a fixed NACA0012 at transonic Mach.

### Run sweep + build dataset

```bash
./scripts/run_sweep.sh
./.venv/bin/python scripts/build_dataset.py --dataset-type aero
```

**Outputs:**
- `datasets/raw/aero_dataset.csv`
- `datasets/processed/aero_ml_dataset.csv`
- `datasets/processed/aero_ml_quality_report.csv`

Environment overrides: `AOA_START`, `AOA_END`, `AOA_STEP`, `MACH_LIST_OVERRIDE`, `REYNOLDS_LIST_OVERRIDE`, `RETRY_MAX`.

### Train models

```bash
./.venv/bin/python ml/train_baseline.py
./.venv/bin/python ml/train_nn.py
```

**Outputs:** `results/baseline_metrics.json`, `results/models/nn_aero_regressor.pt`, `results/nn_metrics.json`

### Full pipeline (one command)

```bash
RUN_TRAINING=1 ./scripts/full_pipeline.sh
```

Logs: `results/logs/full_pipeline_<timestamp>.log`

---

## Pipeline C: Legacy Solidworks NN (Cd / Cl / Cm)

For commercial CFD exports (Solidworks Flow Simulation or similar).

### Step 1 — Generate sampling plan

```bash
cd ml
python train.py --generate-sweep-plan --n-samples 100
```

**Output:** `ml/data/raw/cfd_sweep_plan.csv`

### Step 2 — Run CFD externally

Import the sweep plan into Solidworks. Export results as CSV with columns:

`thickness, camber, position, aoa, reynolds, Cd, Cl, Cm`

Save as `ml/data/raw/cfd_data.csv`.

### Step 3 — Train

```bash
python train.py --data ml/data/raw/cfd_data.csv
```

**Outputs:**
- `ml/results/models/cfd_surrogate_best.pt`
- `ml/results/plots/training_history.png`, `predictions_vs_targets.png`, `error_distribution.png`, `residuals.png`
- `ml/results/metrics.json`, `ml/results/test_predictions.csv`

### Smoke test (no CFD required)

```bash
python ml/generate_demo_data.py
python ml/train.py --data ml/data/raw/cfd_demo_data.csv
```

Demo data is synthetic — suitable for CI, not publication claims.

### Legacy inference

```python
from ml.model import load_model, predict
# Or: inference_engine.CFDExplorer for random-search optimization on Cd/Cl/Cm
```

---

## Module Reference

### `ml/` — Core library

| Script | Purpose | Key CLI flags |
|--------|---------|---------------|
| `train.py` | Legacy PyTorch NN pipeline | `--data`, `--generate-sweep-plan`, `--n-samples`, `--epochs`, `--device` |
| `data_handler.py` | Load CSV, log10(Re), 3σ clip, split, scale | (library) |
| `model.py` | `CFDSurrogateModel`, `Trainer`, `predict`, inference benchmark | (library) |
| `evaluation.py` | MAE, RMSE, MAPE, R²; plot helpers | (library) |
| `train_baseline.py` | Linear + RF on AoA dataset | `--data`, `--test-size`, `--seed` |
| `train_nn.py` | MLP on AoA dataset (optional convergence features) | `--epochs`, `--hidden`, `--inputs`, `--outputs` |
| `train_design_baseline.py` | LR / RF / ET / GP on design dataset | `--engineer-features`, `--smoke-test`, `--min-samples` |
| `validate_optimization.py` | Re-run SU2 on optimized design | `--solver`, `--require-convergence` |
| `metrics_utils.py` | Shared regression metrics | (library) |
| `train_design_nn.py` | Design-space NN (not used by optimizer) | `--epochs`, `--patience`, `--lr` |
| `design_feature_utils.py` | `augment_design_inputs_v1()` | (library) |
| `optimize_design.py` | Bayesian optimization | `--objective`, `--bounds-json`, `--min-cl`, `--max-cd`, `--iters` |
| `interactive_optimize_design.py` | Terminal-driven optimization | optional model path |
| `popup_optimize_design.py` | Tkinter sliders UI | (none) |
| `inference_engine.py` | `CFDExplorer` batch predict + random search | (library) |
| `generate_demo_data.py` | Synthetic LHS data | (none) |
| `run_experiment.py` | Train on demo data | (none) |
| `build_dataset.py` | Alternate merger for `su2_sweep_*` layouts | `--raw-root`, `--latest-only` |

### `scripts/` — SU2 orchestration

| Script | Purpose | Key CLI flags |
|--------|---------|---------------|
| `generate_design_space.py` | LHS/random 6D samples | `--n-samples`, `--sampler`, `--seed`, per-dimension `--*-range` |
| `run_design_sweep.py` | Mesh deform + SU2 per design row | `--solver {rans,euler}`, `--skip-completed`, `--append-raw-output` |
| `build_dataset.py` | AoA or design dataset builder | `--require-strict-convergence`, `--fail-on-rejects` |
| `su2_utils.py` | Convergence checks, coeff parsing | (library) |
| `generate_model_comparison.py` | Consolidated model metrics tables | |
| `generate_research_readiness_report.py` | Weakness / experiment summary | |
| `run_large_design_sweep.sh` | Batch 500+ case workflow | env: `N_SAMPLES`, `SEED`, `SOLVER` |
| `run_sweep.sh` | Multi-AoA sweep on fixed NACA0012 | env: `AOA_*`, `MACH_LIST_OVERRIDE`, `RETRY_MAX` |
| `full_pipeline.sh` | AoA sweep → build → optional train | env: `RUN_TRAINING=1` |

### `webapp/`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Static UI (sliders + objective cards) |
| `/api/health` | GET | Model loaded check |
| `/api/defaults` | GET | Default search bounds |
| `/api/optimize` | POST | Run BO, return best design + CL/CD |

Model path: `MODEL_PATH` env var (default `results/models/design_rf_model.joblib`).

---

## Models, Features, and Metrics

### Model inventory

| Track | Model | Script | Inputs → Outputs | Selection criterion |
|-------|-------|--------|------------------|---------------------|
| Legacy | PyTorch MLP (128→64→32) | `train.py` | 5 geom/flow → Cd, Cl, Cm | Best val loss + early stopping |
| AoA | LinearRegression, RF | `train_baseline.py` | aoa, aoa² → cl, cd | Holdout metrics |
| AoA | MLP (`AeroRegressor`) | `train_nn.py` | aoa + optional convergence → cl, cd | Best val MSE |
| Design | LR, RF(500), ExtraTrees(700), GP | `train_design_baseline.py` | 6D (+7 engineered) → CL, CD | Best mean CV R² |
| Design | NN (96→96→48) | `train_design_nn.py` | 6D → CL, CD | Best val loss |

**Production optimizer loads `design_rf_model.joblib`** (contains the best CV model — often Extra Trees, not always RF).

### Feature engineering

**Legacy:** `log10(reynolds)`, 3σ input clipping, `StandardScaler` on X and y.

**Design base (6D):** `geometry_param_1` (thickness), `geometry_param_2` (camber), `geometry_param_3` (camber position), `AoA`, `Mach`, `Re`.

**Design engineered (+7 with `--engineer-features`):** `sin_AoA`, `cos_AoA`, `log10_Re`, `AoA_x_Mach`, `thick_x_AoA`, `camber_x_camberPos`, `Mach_sq`.

**AoA SU2:** `aoa_squared`, `cl_cd`; optional RMS residual features from `history.csv`.

### Evaluation metrics

| Metric | Used in |
|--------|---------|
| MAE, RMSE, MAPE, R² | Design baseline (`metrics_utils.py`), legacy NN |
| Holdout + RepeatedKFold CV | Design baseline |
| `comparison_table` in metrics JSON | Design baseline → `generate_model_comparison.py` |
| Readiness gate | Design baseline (`PASS` / `FAIL` / `SKIP` in smoke-test mode) |

**Readiness criteria** (`train_design_baseline.py`):
- Mean CV R² ≥ **0.95** per target (CL, CD)
- Mean CV MAE ≤ **5% of training target range** per target (not MAPE)
- Status: `PASS`, `FAIL`, or `SKIP` (small-N smoke test)

**Design quality gate** (`scripts/build_dataset.py`):
- `quality_pass`: `status=success`, physical CL/CD, history present when available
- `strict_pass`: additionally requires `converged=true` and residual thresholds
- Rejected rows logged in `aero_design_quality_report.csv`; dataset still written unless `--fail-on-rejects`

---

## Design Optimization

All UIs call `optimize_design.run_surrogate_optimization()`:

1. Sample initial candidates uniformly in bounds
2. Predict CL/CD via trained surrogate (with feature augmentation if trained with `--engineer-features`)
3. Score by objective with feasibility penalties
4. Fit GP on observed scores; select next point via expected improvement
5. Repeat for `iters` iterations (default 20)

### Objectives

| Objective | Maximizes |
|-----------|-----------|
| `max_cl_cd` | CL / CD (lift-to-drag ratio) |
| `max_cl` | CL |
| `min_cd` | CD (internally negated for maximization) |

### Default constraints (surrogate-side)

- `min_cl` = 0.7 (default)
- `max_cd` = 0.2 (default)

### Default search bounds

| Parameter | Range |
|-----------|-------|
| geometry_thickness | 0.08 – 0.18 |
| geometry_camber | 0.00 – 0.06 |
| geometry_camber_pos | 0.20 – 0.60 |
| aoa (deg) | −2 – 14 |
| mach | 0.65 – 0.82 |
| reynolds | 2×10⁶ – 12×10⁶ |

Override via `--bounds-json` or the web/interactive UIs.

**Important:** Optimization results are surrogate predictions. Always validate with fresh CFD before trusting optima:

```bash
./.venv/bin/python ml/validate_optimization.py
```

At N=44, a measured validation run found the surrogate predicted CL/CD ≈ 1.56/0.038 (L/D ≈ 41) while fresh RANS returned CL ≈ 2.50, CD ≈ −0.06 (non-physical, unconverged) — the optimum was **outside the training hull** on thickness and camber position.

---

## Data Formats and Artifacts

### Design raw CSV (`aero_design_raw.csv`)

Per-row SU2 sweep results: geometry, flow, CL, CD, `status`, `converged`, `failure_reason`, residual features (`rms_rho_final`, `rms_rho_drop`), `case_dir`, `solver`.

### Design processed CSV (`aero_design_dataset.csv`)

Columns: `geometry_param_1`, `geometry_param_2`, `geometry_param_3`, `AoA`, `Mach`, `Re`, `CL`, `CD`

### Legacy CFD CSV

| Column | Description |
|--------|-------------|
| thickness | Max thickness, fraction of chord |
| camber | Max camber, fraction of chord |
| position | Camber position, fraction of chord |
| aoa | Angle of attack, degrees |
| reynolds | Reynolds number |
| Cd, Cl, Cm | Force/moment coefficients |

### Key artifacts (all gitignored)

| Path | Contents |
|------|----------|
| `datasets/raw/` | `design_space.csv`, `aero_design_raw.csv`, `aero_dataset.csv` |
| `datasets/processed/` | `aero_design_dataset.csv`, `aero_design_quality_report.csv`, `aero_ml_*` |
| `results/models/` | `design_rf_model.joblib`, `nn_aero_regressor.pt` |
| `results/` | `design_baseline_metrics.json`, `model_comparison_summary.*`, `optimization_validation.*`, `research_readiness_report.*` |
| `su2_cases/design_runs/` | Per-design SU2 outputs + `sweep_manifest.json` |
| `su2_cases/validation_runs/` | Fresh CFD runs for optimization validation |
| `su2_cases/runs/` | AoA sweep case outputs |
| `ml/results/` | Legacy NN checkpoints and plots |

---

## Configuration

### `ml/config.py` (legacy NN)

| Parameter | Default |
|-----------|---------|
| `INPUT_PARAMS` | thickness, camber, position, aoa, reynolds |
| `OUTPUT_PARAMS` | Cd, Cl, Cm |
| `PARAM_RANGES` | thickness (0.05–0.20), camber (0–0.10), position (0.1–0.5), aoa (−5–15°), reynolds (10⁵–10⁷) |
| `HIDDEN_LAYERS` | [128, 64, 32] |
| `NUM_EPOCHS` | 100 |
| `BATCH_SIZE` | 32 |
| `LEARNING_RATE` | 1e-3 |
| `EARLY_STOPPING_PATIENCE` | 15 |
| `DROPOUT_RATE` | 0.1 |

### SU2 templates

| File | Solver | Notes |
|------|--------|-------|
| `su2_cases/rans_NACA0012.cfg` | `SOLVER= RANS` + Spalart-Allmaras (default) | `REYNOLDS_NUMBER`, `MARKER_HEATFLUX` wall |
| `su2_cases/inv_NACA0012.cfg` | Euler (legacy) | Inviscid fallback via `--solver euler` |

- Default Mach: 0.8
- Parsed outputs: `forces_breakdown.dat` → `Total CL:`, `Total CD:`
- Convergence: `history.csv` RMS density residual thresholds

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Legacy NN, AoA NN, design NN |
| `numpy`, `pandas` | Data handling |
| `scikit-learn` | Baselines, RF surrogate, GP for BO |
| `matplotlib`, `scipy` | Plots, LHS sampling |
| `joblib` | Model serialization |
| `fastapi`, `uvicorn` | Web UI |
| **SU2** (`SU2_CFD`) | CFD solver (system install) |
| **Bash** | Sweep shell scripts |
| **Tkinter** | Desktop popup UI |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `./.venv/bin/python: No such file` | Run from `~/aero-ml-project`; create venv: `python3 -m venv .venv && pip install -r requirements.txt` |
| `KIND_TURB_MODEL must be NONE if SOLVER= NAVIER_STOKES` | Use `rans_NACA0012.cfg` (`SOLVER= RANS`), not `NAVIER_STOKES` + SA |
| `success=0` but SU2 prints Exit Success | Residuals marginal — check `converged` column; use `--no-convergence-check` for debugging only |
| `build_dataset` aborts with rejected rows | Default now excludes rejects and continues; use `--fail-on-rejects` only if you want hard failure |
| `Need at least 8 design samples` | Generate more CFD data, or `--smoke-test` for pipeline checks only |
| Readiness gate FAIL | Expected below ~100–200 cases; scale with `run_large_design_sweep.sh` |
| Validation `CFD CL=None` | Re-run after latest `validate_optimization.py` (captures last parsed coeffs); check `failure_reason` |
| Optimizer finds unrealistic L/D | Surrogate extrapolation — check `within_training_hull` in validation JSON; constrain bounds |
| GP `ConvergenceWarning` during CV | Normal at small N; prefer RF/Extra Trees until dataset grows |
| SU2 cases fail to converge | Narrow bounds in `generate_design_space.py`; increase `--retry-max` |
| Web UI "model not found" | Train first: `ml/train_design_baseline.py`; or set `MODEL_PATH` |
| Duplicate `build_dataset.py` | Use `scripts/build_dataset.py` for SU2 pipelines |

---

## Literature Positioning

See [docs/literature_positioning.md](docs/literature_positioning.md) for how this end-to-end CFD → dataset → train → optimize → validate pipeline differs from SMT, Dakota, and generic standalone ML workflows.

## References

**Surrogate modeling**
- Forrester et al., 2008 — *Engineering Design via Surrogate Modelling*
- Queipo et al., 2005 — *Surrogate-based analysis and optimization*

**ML for aerodynamics**
- Chen et al., 2021 — *Machine learning for aerodynamic design*
- Sekar et al., 2019 — *FastSurferCNN*

**Physics-informed ML**
- Raissi et al., 2019 — *Physics-informed neural networks*

**Aerodynamic theory**
- Abbott & Von Doenhoff, 1959 — *Theory of wing sections*
- Anderson, 2011 — *Fundamentals of aerodynamics*

**Software**
- [SU2](https://su2code.github.io/) — open-source CFD suite
