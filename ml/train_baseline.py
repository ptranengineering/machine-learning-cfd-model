#!/usr/bin/env python3
"""
Baseline regressors for aerodynamic coefficient prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "datasets" / "processed" / "aero_ml_dataset.csv"
DEFAULT_RESULTS = ROOT / "results" / "baseline_metrics.json"


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test) -> dict:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    out = {"model": name}
    targets = ["cl", "cd"]
    for i, target in enumerate(targets):
        rmse = float(np.sqrt(mean_squared_error(y_test[:, i], pred[:, i])))
        r2 = None if len(y_test) < 2 else float(r2_score(y_test[:, i], pred[:, i]))
        out[target] = {"rmse": rmse, "r2": r2}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline aerodynamic regressors.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--test-size", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    required = ["aoa", "aoa_squared", "cl", "cd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    x = df[["aoa", "aoa_squared"]].to_numpy(dtype=float)
    y = df[["cl", "cd"]].to_numpy(dtype=float)

    if len(df) < 3:
        raise ValueError("Need at least 3 samples to train/evaluate baseline models.")

    dynamic_test_size = args.test_size
    if int(round(len(df) * dynamic_test_size)) < 2:
        dynamic_test_size = min(0.5, max(args.test_size, 2 / len(df)))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=dynamic_test_size, random_state=args.seed
    )

    models = [
        ("linear_regression", LinearRegression()),
        ("random_forest", MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=args.seed))),
    ]

    metrics = {"n_samples": int(len(df)), "results": []}
    for name, model in models:
        metrics["results"].append(evaluate_model(name, model, x_train, x_test, y_train, y_test))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] Baseline metrics written to {args.output}")
    for row in metrics["results"]:
        cl_r2 = "n/a" if row["cl"]["r2"] is None else f"{row['cl']['r2']:.4f}"
        cd_r2 = "n/a" if row["cd"]["r2"] is None else f"{row['cd']['r2']:.4f}"
        print(f"[{row['model']}] CL rmse={row['cl']['rmse']:.6f} r2={cl_r2} | CD rmse={row['cd']['rmse']:.6f} r2={cd_r2}")


if __name__ == "__main__":
    main()
