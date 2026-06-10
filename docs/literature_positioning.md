# Literature Positioning: Aero-ML vs. Established Surrogate Frameworks

This document summarizes how the **aero-ml-project** end-to-end pipeline differs from common surrogate-modeling tools and generic ML workflows.

## What This Framework Provides

An automated loop:

```
Design sampling (LHS) → SU2 CFD (mesh deformation, convergence checks)
    → Quality-filtered dataset → Surrogate training (LR / RF / GP / NN)
    → Bayesian optimization → Optional SU2 re-validation
```

Key differentiators:

- **CFD-native orchestration** — parametric NACA mesh deformation, per-case manifests, residual-based acceptance
- **Physics-aware filtering** — convergence residuals, coefficient bounds, reproducible quality reports
- **Surrogate comparison built-in** — cross-validated metrics (MAE, RMSE, R², relative % error) across model families
- **Closed-loop validation** — `ml/validate_optimization.py` re-runs SU2 on optimized designs

## Comparison to SMT (Surrogate Modeling Toolbox)

| Aspect | SMT | This framework |
|--------|-----|----------------|
| Scope | Surrogate fitting library (Kriging, RBF, IDW, etc.) | Full CFD → dataset → train → optimize pipeline |
| CFD coupling | User supplies `(X, y)` | Generates `y` via SU2 sweeps |
| Geometry | External | Parametric airfoil deformation on SU2 mesh |
| Quality control | User responsibility | Automated convergence + coefficient gates |
| Optimization | Not included | Bayesian optimization on trained surrogate |

**When to use SMT:** You already have a CFD database and need advanced surrogate algorithms.

**When to use this repo:** You need to *produce* the database from SU2 and run design optimization with validation.

## Comparison to Dakota

| Aspect | Dakota | This framework |
|--------|--------|----------------|
| Role | UQ / optimization orchestrator | CFD-ML integration for aerodynamic design |
| Solver interface | Generic analysis drivers | Native SU2 cfg/mesh workflow |
| ML surrogates | Interfaces to external models | In-repo training (sklearn, PyTorch) |
| Mesh deformation | External | Built-in NACA thickness/camber deformation |
| Target domain | General engineering UQ | Transonic airfoil CL/CD surrogates |

**When to use Dakota:** Multi-disciplinary UQ, sampling studies with existing analysis codes.

**When to use this repo:** Rapid iteration on airfoil design-space surrogates with SU2 as the sole physics engine.

## Comparison to Generic Standalone ML Workflows

Typical notebook workflow:

1. Export CFD CSV manually
2. Train a model in isolation
3. Optimize with arbitrary bounds
4. No automatic verification

| Gap in generic ML | This framework |
|-------------------|----------------|
| No CFD automation | `run_design_sweep.py` + `run_large_design_sweep.sh` |
| No convergence filtering | Residual checks in sweep + `aero_design_quality_report.csv` |
| No reproducibility manifest | `sweep_manifest.json` per run |
| Surrogate-only optima | `validate_optimization.py` |
| Single model | Multi-model CV selection + comparison tables |

## Reproducibility Features

- Per-run manifest: cfg hash, SU2 version, seed, convergence thresholds
- Quality reports: accepted vs. rejected cases with reason flags
- Metrics JSON: holdout + repeated K-fold CV with readiness gate
- Gitignored artifacts with documented regeneration commands

## Suggested Citation Framing

> An end-to-end aerodynamic surrogate pipeline coupling automated SU2 RANS dataset generation, multi-model regression benchmarks, and Bayesian optimization with optional CFD re-validation—bridging specialized surrogate libraries (SMT) and general UQ frameworks (Dakota) with a CFD-native ML workflow.
