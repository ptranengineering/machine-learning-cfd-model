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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "datasets" / "processed" / "aero_design_dataset.csv"
DEFAULT_METRICS = ROOT / "results" / "design_baseline_metrics.json"
DEFAULT_MODEL = ROOT / "results" / "models" / "design_rf_model.joblib"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def evaluate(name: str, model, x_train, x_test, y_train, y_test) -> dict:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    out = {"model": name, "targets": {}}
    for i, target in enumerate(["CL", "CD"]):
        out["targets"][target] = {
            "rmse": rmse(y_test[:, i], pred[:, i]),
            "r2": float(r2_score(y_test[:, i], pred[:, i])) if len(y_test) > 1 else None,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train design-space baseline regressors.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
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

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed
    )

    lin = LinearRegression()
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=args.seed))

    metrics = {
        "n_samples": int(len(df)),
        "features": feature_cols,
        "targets": target_cols,
        "results": [
            evaluate("linear_regression", lin, x_train, x_test, y_train, y_test),
            evaluate("random_forest", rf, x_train, x_test, y_train, y_test),
        ],
    }

    # Persist the stronger nonlinear baseline for downstream optimization.
    rf.fit(x, y)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "feature_cols": feature_cols, "target_cols": target_cols}, args.model_out)

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] metrics -> {args.metrics}")
    print(f"[DONE] model -> {args.model_out}")


if __name__ == "__main__":
    main()
