#!/usr/bin/env python3
"""
Train multivariate baseline models for design-level CL/CD prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "datasets" / "processed" / "aero_design_dataset.csv"
DEFAULT_METRICS = ROOT / "results" / "design_baseline_metrics.json"
DEFAULT_MODEL = ROOT / "results" / "models" / "design_rf_model.joblib"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def evaluate(model, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, target_cols: list[str]) -> dict:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    out = {"targets": {}}
    for i, target in enumerate(target_cols):
        out["targets"][target] = {
            "rmse": rmse(y_test[:, i], pred[:, i]),
            "mae": float(mean_absolute_error(y_test[:, i], pred[:, i])),
            "r2": float(r2_score(y_test[:, i], pred[:, i])) if len(y_test) > 1 else None,
        }
    return out


def summarize_cv_folds(folds: list[dict], target_cols: list[str]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for target in target_cols:
        summary[target] = {}
        for metric in ("r2", "rmse", "mae"):
            vals = [f["targets"][target][metric] for f in folds]
            summary[target][f"{metric}_mean"] = float(np.mean(vals))
            summary[target][f"{metric}_std"] = float(np.std(vals))
    return summary


def cross_validate(model, x: np.ndarray, y: np.ndarray, target_cols: list[str], seed: int, n_splits: int, n_repeats: int) -> dict:
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds: list[dict] = []
    for train_idx, test_idx in rkf.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        folds.append(evaluate(model, x_train, x_test, y_train, y_test, target_cols))
    return {"folds": int(len(folds)), "targets": summarize_cv_folds(folds, target_cols)}


def main() -> None:
    p = argparse.ArgumentParser(description="Train design-space baseline regressors.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    feature_cols = [
        "geometry_param_1",
        "geometry_param_2",
        "geometry_param_3",
        "AoA",
        "Mach",
        "Re",
    ]
    target_cols = ["CL", "CD"]
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df[feature_cols].to_numpy(dtype=float)
    y = df[target_cols].to_numpy(dtype=float)

    if len(df) < 8:
        raise ValueError("Need at least 8 design samples to train baseline reliably.")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.test_size, random_state=args.seed)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=500, random_state=args.seed)),
        "extra_trees": MultiOutputRegressor(ExtraTreesRegressor(n_estimators=700, random_state=args.seed)),
    }

    holdout_results = []
    cv_results = []
    for model_name, model in models.items():
        holdout_results.append(
            {
                "model": model_name,
                **evaluate(model, x_train, x_test, y_train, y_test, target_cols),
            }
        )
        cv_results.append(
            {
                "model": model_name,
                **cross_validate(model, x, y, target_cols, args.seed, args.cv_splits, args.cv_repeats),
            }
        )

    best_cv = max(
        cv_results,
        key=lambda r: (r["targets"]["CL"]["r2_mean"] + r["targets"]["CD"]["r2_mean"]) / 2.0,
    )

    target_ranges = {
        "CL": float(np.max(y[:, 0]) - np.min(y[:, 0])),
        "CD": float(np.max(y[:, 1]) - np.min(y[:, 1])),
    }
    readiness_criteria = {
        "min_r2_mean_each_target": 0.95,
        "max_mae_pct_of_range_each_target": 5.0,
    }
    readiness_checks = {}
    for target in target_cols:
        mae_mean = best_cv["targets"][target]["mae_mean"]
        readiness_checks[target] = {
            "r2_mean": best_cv["targets"][target]["r2_mean"],
            "mae_pct_of_range": float(100.0 * mae_mean / max(target_ranges[target], 1e-12)),
        }

    readiness_pass = all(
        readiness_checks[target]["r2_mean"] >= readiness_criteria["min_r2_mean_each_target"]
        and readiness_checks[target]["mae_pct_of_range"] <= readiness_criteria["max_mae_pct_of_range_each_target"]
        for target in target_cols
    )

    metrics = {
        "n_samples": int(len(df)),
        "features": feature_cols,
        "targets": target_cols,
        "holdout_split": {"test_size": args.test_size, "seed": args.seed},
        "cv_strategy": {
            "type": "RepeatedKFold",
            "n_splits": args.cv_splits,
            "n_repeats": args.cv_repeats,
            "seed": args.seed,
        },
        "holdout_results": holdout_results,
        "cross_validation_results": cv_results,
        "selected_model": best_cv["model"],
        "readiness": {
            "criteria": readiness_criteria,
            "checks": readiness_checks,
            "status": "PASS" if readiness_pass else "FAIL",
        },
    }

    selected_model = models[best_cv["model"]]
    selected_model.fit(x, y)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": selected_model, "feature_cols": feature_cols, "target_cols": target_cols}, args.model_out)

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] metrics -> {args.metrics}")
    print(f"[DONE] model -> {args.model_out}")


if __name__ == "__main__":
    main()
