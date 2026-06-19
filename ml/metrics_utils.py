"""
Shared regression metrics for surrogate model evaluation.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    yt = y_true[:, 0] if y_true.ndim == 2 and y_true.shape[1] == 1 else y_true
    yp = y_pred[:, 0] if y_pred.ndim == 2 and y_pred.shape[1] == 1 else y_pred
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else None
    rel_pct = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100.0)
    return {
        "target": target,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": rel_pct,
        "relative_pct_error": rel_pct,
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    per_target = []
    for i, name in enumerate(target_names):
        per_target.append(
            per_target_metrics(y_true[:, i : i + 1], y_pred[:, i : i + 1], name)
        )

    r2_vals = [t["r2"] for t in per_target if t["r2"] is not None]
    aggregate = {
        "mae": float(np.mean([t["mae"] for t in per_target])),
        "rmse": float(np.mean([t["rmse"] for t in per_target])),
        "r2_mean": float(np.mean(r2_vals)) if r2_vals else None,
        "mape": float(np.mean([t["mape"] for t in per_target])),
        "relative_pct_error": float(np.mean([t["relative_pct_error"] for t in per_target])),
    }
    return {"per_target": per_target, "aggregate": aggregate}


def benchmark_inference_latency(model, X_sample: np.ndarray, n_repeats: int = 1000) -> dict:
    """Wall-clock latency for a single surrogate prediction (milliseconds)."""
    x = np.asarray(X_sample, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    for _ in range(10):
        model.predict(x)

    times_ms: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict(x)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=float)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "n_repeats": int(n_repeats),
    }
