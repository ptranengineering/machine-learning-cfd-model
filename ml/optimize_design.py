#!/usr/bin/env python3
"""
Bayesian optimization over design parameters using a trained surrogate.
Objectives: maximize CL/CD, maximize CL, or minimize CD (with optional CL/CD box constraints).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from design_feature_utils import augment_design_inputs_v1


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "results" / "models" / "design_rf_model.joblib"
DEFAULT_OUT = ROOT / "results" / "design_optimization_result.json"

# Physical order: thickness, camber, camber_position, AoA (deg), Mach, Reynolds
RANGE_PARAM_NAMES = (
    "geometry_thickness",
    "geometry_camber",
    "geometry_camber_pos",
    "aoa",
    "mach",
    "reynolds",
)

Objective = Literal["max_cl_cd", "max_cl", "min_cd"]


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    z = (mu - best) / (sigma + 1e-9)
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + np.tanh(np.sqrt(np.pi / 8.0) * z))
    return (mu - best) * Phi + sigma * phi


def default_bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([0.08, 0.00, 0.20, -2.0, 0.65, 2.0e6], dtype=float)
    hi = np.array([0.18, 0.06, 0.60, 14.0, 0.82, 12.0e6], dtype=float)
    return lo, hi


def sample_candidates(n: int, seed: int, lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> np.ndarray:
    """Uniform samples in [lo, hi]^6 (one row per candidate)."""
    if lo is None or hi is None:
        lo, hi = default_bounds()
    if lo.shape != (6,) or hi.shape != (6,) or np.any(hi <= lo):
        raise ValueError("Bounds must be length-6 arrays with hi > lo for each dimension.")
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=(n, 6))
    return lo + (hi - lo) * u


def augment_if_needed(base_x: np.ndarray, augment_version: int) -> np.ndarray:
    if augment_version <= 0:
        return base_x
    aug_x, _ = augment_design_inputs_v1(base_x)
    return aug_x


def _objective_vectors(
    cl: np.ndarray,
    cd: np.ndarray,
    objective: Objective,
    min_cl: float,
    max_cd: float,
) -> np.ndarray:
    cd_safe = np.maximum(cd, 1e-12)
    feas = (cl >= min_cl) & (cd_safe <= max_cd) & (cd_safe > 0)
    if objective == "max_cl_cd":
        return np.where(feas, cl / cd_safe, -1e6)
    if objective == "max_cl":
        return np.where(feas, cl, -1e6)
    if objective == "min_cd":
        return np.where(feas, -cd_safe, -1e6)
    raise ValueError(f"Unknown objective {objective!r}")


def _scalar_objective(cl: float, cd: float, objective: Objective, min_cl: float, max_cd: float) -> float:
    return float(
        _objective_vectors(
            np.array([cl], dtype=float),
            np.array([cd], dtype=float),
            objective,
            min_cl,
            max_cd,
        )[0]
    )


def run_surrogate_optimization(
    pack: dict[str, Any],
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    objective: Objective = "max_cl_cd",
    min_cl: float = 0.7,
    max_cd: float = 0.2,
    iters: int = 20,
    init_samples: int = 25,
    candidate_pool: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Core BO loop on the surrogate; maximizes internal score (min_cd uses negative CD)."""
    surrogate = pack["model"]
    augment_version = int(pack.get("augment_version", 0))

    x_obs = sample_candidates(init_samples, seed, lo, hi)
    pred = surrogate.predict(augment_if_needed(x_obs, augment_version))
    y_obj = _objective_vectors(pred[:, 0], pred[:, 1], objective, min_cl, max_cd)

    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)

    for t in range(iters):
        gpr.fit(x_obs, y_obj)
        pool = sample_candidates(candidate_pool, seed + t + 1, lo, hi)
        mu, std = gpr.predict(pool, return_std=True)
        ei = expected_improvement(mu, std, float(np.max(y_obj)))
        x_next = pool[int(np.argmax(ei))]

        pred_next = surrogate.predict(augment_if_needed(x_next.reshape(1, -1), augment_version))[0]
        cl_n, cd_n = float(pred_next[0]), float(pred_next[1])
        y_next = _scalar_objective(cl_n, cd_n, objective, min_cl, max_cd)

        x_obs = np.vstack([x_obs, x_next])
        y_obj = np.append(y_obj, y_next)

    best_idx = int(np.argmax(y_obj))
    best_x = x_obs[best_idx]
    best_pred = surrogate.predict(augment_if_needed(best_x.reshape(1, -1), augment_version))[0]
    cl_b, cd_b = float(best_pred[0]), float(best_pred[1])

    obj_label = {
        "max_cl_cd": "maximize CL/CD",
        "max_cl": "maximize CL",
        "min_cd": "minimize CD",
    }[objective]

    return {
        "objective": obj_label,
        "objective_key": objective,
        "search_bounds": {
            name: [float(lo[i]), float(hi[i])] for i, name in enumerate(RANGE_PARAM_NAMES)
        },
        "constraints": {"min_cl": min_cl, "max_cd": max_cd},
        "best_geometry_flow": {
            "geometry_thickness": float(best_x[0]),
            "geometry_camber": float(best_x[1]),
            "geometry_camber_pos": float(best_x[2]),
            "aoa": float(best_x[3]),
            "mach": float(best_x[4]),
            "reynolds": float(best_x[5]),
        },
        "predicted": {
            "CL": cl_b,
            "CD": cd_b,
            "CL_CD": (cl_b / cd_b) if cd_b != 0 else None,
        },
        "best_internal_score": float(np.max(y_obj)),
        "iterations": iters,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Bayesian optimize design using surrogate.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--init-samples", type=int, default=25)
    p.add_argument("--candidate-pool", type=int, default=500)
    p.add_argument("--min-cl", type=float, default=0.7)
    p.add_argument("--max-cd", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--objective",
        choices=["max_cl_cd", "max_cl", "min_cd"],
        default="max_cl_cd",
        help="What to optimize using the surrogate (constraints still apply).",
    )
    p.add_argument(
        "--bounds-json",
        type=Path,
        default=None,
        help='JSON file: {"geometry_thickness":[lo,hi], ...} for all six keys; otherwise use script defaults.',
    )
    args = p.parse_args()

    lo, hi = default_bounds()
    if args.bounds_json is not None:
        data = json.loads(args.bounds_json.read_text(encoding="utf-8"))
        for i, name in enumerate(RANGE_PARAM_NAMES):
            pair = data.get(name)
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError(f'bounds-json missing or invalid key "{name}" (need [lo, hi])')
            lo[i], hi[i] = float(pair[0]), float(pair[1])

    pack = joblib.load(args.model)
    result = run_surrogate_optimization(
        pack,
        lo,
        hi,
        objective=args.objective,
        min_cl=args.min_cl,
        max_cd=args.max_cd,
        iters=args.iters,
        init_samples=args.init_samples,
        candidate_pool=args.candidate_pool,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[DONE] optimization result -> {args.output}")
    print(f"[INFO] best_internal_score={result['best_internal_score']:.6f}")


if __name__ == "__main__":
    main()
