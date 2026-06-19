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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor

from design_feature_utils import BASE_COLS, augment_design_inputs_v1
from figure_utils import FIG_DIR, plot_feature_importance, plot_parity_plots
from metrics_utils import benchmark_inference_latency, compute_regression_metrics


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "datasets" / "processed" / "aero_design_dataset.csv"
DEFAULT_METRICS = ROOT / "results" / "design_baseline_metrics.json"
DEFAULT_MODEL = ROOT / "results" / "models" / "design_rf_model.joblib"
FIG3_PATH = FIG_DIR / "fig3_parity_plots.png"
FIG4_PATH = FIG_DIR / "fig4_feature_importance.png"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def make_gp_model(seed: int) -> MultiOutputRegressor:
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    return MultiOutputRegressor(
        GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            alpha=1e-6,
            n_restarts_optimizer=2,
            random_state=seed,
        )
    )


def evaluate(model, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, target_cols: list[str]) -> dict:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = compute_regression_metrics(y_test, pred, target_cols)
    out = {"targets": {}}
    for entry in metrics["per_target"]:
        target = entry["target"]
        out["targets"][target] = {
            "rmse": entry["rmse"],
            "mae": entry["mae"],
            "r2": entry["r2"],
            "mape": entry["mape"],
            "relative_pct_error": entry["relative_pct_error"],
        }
    out["aggregate"] = metrics["aggregate"]
    return out


def summarize_cv_folds(folds: list[dict], target_cols: list[str]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for target in target_cols:
        summary[target] = {}
        for metric in ("r2", "rmse", "mae", "mape", "relative_pct_error"):
            vals = [f["targets"][target][metric] for f in folds if f["targets"][target].get(metric) is not None]
            if vals:
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
    p.add_argument(
        "--engineer-features",
        action="store_true",
        help=f"Augment BASE inputs with nonlinear transforms ({', '.join(BASE_COLS)} -> +extras). "
        "If enabled, saved model metadata records augment_version=1.",
    )
    p.add_argument(
        "--save-figures",
        action="store_true",
        help="Save Fig. 3 (parity plots) and Fig. 4 (feature importance) to results/figures/.",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=8,
        help="Minimum rows required unless --smoke-test is set.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Allow training on very small datasets (metrics/readiness not meaningful).",
    )
    args = p.parse_args()

    df = pd.read_csv(args.data)
    feature_cols = list(BASE_COLS)
    target_cols = ["CL", "CD"]
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df[BASE_COLS].to_numpy(dtype=float)
    augment_version = 1 if args.engineer_features else 0
    if args.engineer_features:
        x, feature_cols = augment_design_inputs_v1(x)
    y = df[target_cols].to_numpy(dtype=float)

    n_samples = len(df)
    if n_samples < 1:
        raise ValueError("Dataset is empty.")
    if n_samples < args.min_samples and not args.smoke_test:
        raise ValueError(
            f"Need at least {args.min_samples} design samples to train baseline reliably "
            f"(found {n_samples}). Generate more CFD data, or pass --smoke-test for pipeline checks."
        )

    small_n_mode = args.smoke_test or n_samples < args.min_samples
    if small_n_mode:
        print(
            f"[WARN] small-n training mode ({n_samples} samples): "
            "holdout/CV metrics and readiness gate are not meaningful."
        )

    models: dict = {
        "linear_regression": LinearRegression(),
        "random_forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=500, random_state=args.seed)),
    }
    if not small_n_mode:
        models["extra_trees"] = MultiOutputRegressor(
            ExtraTreesRegressor(n_estimators=700, random_state=args.seed)
        )
        models["gaussian_process"] = make_gp_model(args.seed)
    elif n_samples >= 4:
        models["gaussian_process"] = make_gp_model(args.seed)

    holdout_results: list[dict] = []
    cv_results: list[dict] = []
    cv_splits = min(args.cv_splits, n_samples) if n_samples >= 2 else 1

    if small_n_mode:
        x_train, x_test, y_train, y_test = x, x, y, y
        for model_name, model in models.items():
            holdout_results.append(
                {
                    "model": model_name,
                    **evaluate(model, x_train, x_test, y_train, y_test, target_cols),
                }
            )
        best_cv = {"model": "random_forest", "targets": {}}
        for target in target_cols:
            best_cv["targets"][target] = {"r2_mean": None, "mae_mean": None}
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=args.test_size, random_state=args.seed
        )
        for model_name, model in models.items():
            holdout_results.append(
                {
                    "model": model_name,
                    **evaluate(model, x_train, x_test, y_train, y_test, target_cols),
                }
            )
            if n_samples >= cv_splits and cv_splits >= 2:
                cv_results.append(
                    {
                        "model": model_name,
                        **cross_validate(
                            model, x, y, target_cols, args.seed, cv_splits, args.cv_repeats
                        ),
                    }
                )
        if cv_results:
            best_cv = max(
                cv_results,
                key=lambda r: (r["targets"]["CL"]["r2_mean"] + r["targets"]["CD"]["r2_mean"]) / 2.0,
            )
        else:
            best_cv = holdout_results[0]

    target_ranges = {
        "CL": float(np.max(y[:, 0]) - np.min(y[:, 0])),
        "CD": float(np.max(y[:, 1]) - np.min(y[:, 1])),
    }
    readiness_criteria = {
        "min_r2_mean_each_target": 0.95,
        "max_mae_pct_of_range_each_target": 5.0,
    }
    readiness_checks = {}
    if small_n_mode or not cv_results:
        readiness_checks = {
            target: {"r2_mean": None, "mae_pct_of_range": None} for target in target_cols
        }
        readiness_pass = False
        readiness_status = "SKIP"
    else:
        for target in target_cols:
            mae_mean = best_cv["targets"][target]["mae_mean"]
            readiness_checks[target] = {
                "r2_mean": best_cv["targets"][target]["r2_mean"],
                "mae_pct_of_range": float(100.0 * mae_mean / max(target_ranges[target], 1e-12)),
            }
        readiness_pass = all(
            readiness_checks[target]["r2_mean"] >= readiness_criteria["min_r2_mean_each_target"]
            and readiness_checks[target]["mae_pct_of_range"]
            <= readiness_criteria["max_mae_pct_of_range_each_target"]
            for target in target_cols
        )
        readiness_status = "PASS" if readiness_pass else "FAIL"

    comparison_table = []
    for hr in holdout_results:
        for target in target_cols:
            t = hr["targets"][target]
            comparison_table.append(
                {
                    "model": hr["model"],
                    "target": target,
                    "split": "holdout",
                    "mae": t["mae"],
                    "rmse": t["rmse"],
                    "r2": t["r2"],
                    "mape": t.get("mape"),
                    "relative_pct_error": t.get("relative_pct_error"),
                }
            )
    for cr in cv_results:
        for target in target_cols:
            t = cr["targets"][target]
            comparison_table.append(
                {
                    "model": cr["model"],
                    "target": target,
                    "split": "cv_mean",
                    "mae": t.get("mae_mean"),
                    "rmse": t.get("rmse_mean"),
                    "r2": t.get("r2_mean"),
                    "mape": t.get("mape_mean"),
                    "relative_pct_error": t.get("relative_pct_error_mean"),
                }
            )

    selected_name = best_cv["model"] if isinstance(best_cv.get("model"), str) else "random_forest"

    metrics = {
        "n_samples": int(n_samples),
        "features": feature_cols,
        "targets": target_cols,
        "small_n_mode": small_n_mode,
        "holdout_split": {"test_size": args.test_size, "seed": args.seed},
        "cv_strategy": {
            "type": "RepeatedKFold" if cv_results else "skipped",
            "n_splits": cv_splits,
            "n_repeats": args.cv_repeats,
            "seed": args.seed,
        },
        "holdout_results": holdout_results,
        "cross_validation_results": cv_results,
        "comparison_table": comparison_table,
        "selected_model": selected_name,
        "readiness": {
            "criteria": readiness_criteria,
            "checks": readiness_checks,
            "status": readiness_status,
        },
    }

    selected_model = models[selected_name]
    selected_model.fit(x, y)

    rng = np.random.default_rng(args.seed)
    latency = benchmark_inference_latency(selected_model, rng.random((1, x.shape[1])), n_repeats=1000)

    bo_time_s: float | None = None
    if not small_n_mode:
        import time

        from optimize_design import default_bounds, run_surrogate_optimization

        pack = {
            "model": selected_model,
            "augment_version": augment_version,
        }
        lo, hi = default_bounds()
        t0 = time.perf_counter()
        run_surrogate_optimization(pack, lo, hi, iters=20, seed=args.seed)
        bo_time_s = time.perf_counter() - t0

    metrics["latency_ms"] = {
        "inference": latency,
        "bayesian_optimization_20_iter_s": bo_time_s,
    }

    if args.save_figures:
        rf_model = models["random_forest"]
        rf_model.fit(x_train, y_train)
        plot_parity_plots(
            x_test, y_test, rf_model, feature_cols, target_cols, FIG3_PATH,
            model_label="RF + engineered features" if args.engineer_features else "RF",
        )
        plot_feature_importance(rf_model, x_test, y_test, feature_cols, target_cols, FIG4_PATH, seed=args.seed)
        print(f"[DONE] figures -> {FIG3_PATH}, {FIG4_PATH}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": selected_model,
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "augment_version": augment_version,
        },
        args.model_out,
    )

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] metrics -> {args.metrics}")
    print(f"[DONE] model -> {args.model_out}")

    best_r2 = None
    if cv_results:
        best_r2 = (best_cv["targets"]["CL"]["r2_mean"] + best_cv["targets"]["CD"]["r2_mean"]) / 2.0
    tier = "Exploratory"
    if readiness_status == "PASS":
        tier = "Certification"
    elif best_r2 is not None and best_r2 >= 0.80:
        tier = "Design"
    r2_str = f"{best_r2:.4f}" if best_r2 is not None else "n/a"
    print(
        f"[SUMMARY] N={n_samples} | best CV R²(avg)={r2_str} "
        f"| readiness={readiness_status} | tier={tier} "
        f"| inference={latency['mean_ms']:.3f}±{latency['std_ms']:.3f} ms"
    )


if __name__ == "__main__":
    main()
