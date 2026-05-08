#!/usr/bin/env python3
"""
Bayesian optimization over design parameters using a trained surrogate.
Objective: maximize CL/CD with CL and CD constraints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "results" / "models" / "design_rf_model.joblib"
DEFAULT_OUT = ROOT / "results" / "design_optimization_result.json"


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    # Lightweight EI approximation without scipy dependency.
    z = (mu - best) / (sigma + 1e-9)
    # Approximate Phi and phi with tanh-based CDF and Gaussian pdf.
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + np.tanh(np.sqrt(np.pi / 8.0) * z))
    return (mu - best) * Phi + sigma * phi


def sample_candidates(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # [thickness, camber, camber_pos, AoA, Mach, Re]
    lo = np.array([0.08, 0.00, 0.20, -2.0, 0.65, 2.0e6], dtype=float)
    hi = np.array([0.18, 0.06, 0.60, 14.0, 0.82, 12.0e6], dtype=float)
    u = rng.uniform(0.0, 1.0, size=(n, 6))
    return lo + (hi - lo) * u


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
    args = p.parse_args()

    pack = joblib.load(args.model)
    surrogate = pack["model"]

    x_obs = sample_candidates(args.init_samples, args.seed)
    pred = surrogate.predict(x_obs)
    cl = pred[:, 0]
    cd = pred[:, 1]
    feasible = (cl >= args.min_cl) & (cd <= args.max_cd) & (cd > 0)
    y_obj = np.where(feasible, cl / cd, -1e6)

    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=args.seed)

    for t in range(args.iters):
        gpr.fit(x_obs, y_obj)
        pool = sample_candidates(args.candidate_pool, args.seed + t + 1)
        mu, std = gpr.predict(pool, return_std=True)
        ei = expected_improvement(mu, std, float(np.max(y_obj)))
        x_next = pool[int(np.argmax(ei))]

        pred_next = surrogate.predict(x_next.reshape(1, -1))[0]
        cl_n, cd_n = float(pred_next[0]), float(pred_next[1])
        feasible_n = cl_n >= args.min_cl and cd_n <= args.max_cd and cd_n > 0
        y_next = (cl_n / cd_n) if feasible_n else -1e6

        x_obs = np.vstack([x_obs, x_next])
        y_obj = np.append(y_obj, y_next)

    best_idx = int(np.argmax(y_obj))
    best_x = x_obs[best_idx]
    best_pred = surrogate.predict(best_x.reshape(1, -1))[0]
    result = {
        "objective": "maximize CL/CD",
        "constraints": {"min_cl": args.min_cl, "max_cd": args.max_cd},
        "best_geometry_flow": {
            "geometry_thickness": float(best_x[0]),
            "geometry_camber": float(best_x[1]),
            "geometry_camber_pos": float(best_x[2]),
            "aoa": float(best_x[3]),
            "mach": float(best_x[4]),
            "reynolds": float(best_x[5]),
        },
        "predicted": {
            "CL": float(best_pred[0]),
            "CD": float(best_pred[1]),
            "CL_CD": float(best_pred[0] / best_pred[1]) if best_pred[1] != 0 else None,
        },
        "best_objective": float(np.max(y_obj)),
        "iterations": args.iters,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[DONE] optimization result -> {args.output}")
    print(f"[INFO] best objective={result['best_objective']:.4f}")


if __name__ == "__main__":
    main()
