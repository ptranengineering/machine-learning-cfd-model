#!/usr/bin/env python3
"""
Learning-curve experiment: CV metrics vs. training set size N.

Saves results/learning_curve_data.json and optionally Fig. 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from design_feature_utils import BASE_COLS, augment_design_inputs_v1
from figure_utils import (
    C_CERT_GATE,
    C_DESIGN_GATE,
    C_ET,
    C_RF,
    FIG_DIR,
    apply_paper_style,
    mae_pct_of_range,
    panel_label,
    save_figure,
    style_axes,
)
from train_design_baseline import cross_validate

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "datasets" / "processed" / "aero_design_dataset.csv"
DEFAULT_JSON = ROOT / "results" / "learning_curve_data.json"
DEFAULT_FIG = FIG_DIR / "fig2_learning_curves.png"

TARGET_NS = [50, 100, 150, 200, 300, 400, 500]
MODELS = {
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
}


def _make_models(seed: int) -> dict:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    return {
        "random_forest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=500, random_state=seed)
        ),
        "extra_trees": MultiOutputRegressor(
            ExtraTreesRegressor(n_estimators=700, random_state=seed)
        ),
    }


def _project_value(ns_measured: list[int], vals: list[float], n_target: int) -> float:
    """Log-scale extrapolation from the last two measured points."""
    if not ns_measured:
        return float("nan")
    if n_target <= ns_measured[-1]:
        for n, v in zip(ns_measured, vals):
            if n == n_target:
                return float(v)
    if len(ns_measured) < 2:
        return float(vals[-1])
    n1, n2 = ns_measured[-2], ns_measured[-1]
    v1, v2 = vals[-2], vals[-1]
    if v1 <= 0 or v2 <= 0 or n1 <= 0 or n2 <= 0:
        slope = (v2 - v1) / max(n2 - n1, 1)
        return float(v2 + slope * (n_target - n2))
    log_slope = (np.log(v2) - np.log(v1)) / (np.log(n2) - np.log(n1))
    return float(np.exp(np.log(v2) + log_slope * (np.log(n_target) - np.log(n2))))


def run_experiment(
    x: np.ndarray,
    y: np.ndarray,
    target_cols: list[str],
    ns: list[int],
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> dict:
    n_total = len(x)
    rng = np.random.default_rng(seed)
    measured: dict[str, dict] = {key: {} for key in MODELS}

    measure_ns = sorted(
        {
            n
            for n in list(ns) + [n_total]
            if n_splits <= n <= n_total
        }
    )

    for n in measure_ns:
        n_eff = min(n, n_total)
        idx = rng.choice(n_total, size=n_eff, replace=False)
        x_sub, y_sub = x[idx], y[idx]

        for model_key in MODELS:
            model = _make_models(seed)[model_key]
            cv = cross_validate(model, x_sub, y_sub, target_cols, seed, n_splits, n_repeats)
            entry: dict[str, dict[str, float]] = {}
            for target in target_cols:
                t = cv["targets"][target]
                mae_mean = t["mae_mean"]
                y_range = float(
                    np.max(y_sub[:, target_cols.index(target)])
                    - np.min(y_sub[:, target_cols.index(target)])
                )
                entry[target] = {
                    "r2_mean": float(t["r2_mean"]),
                    "r2_std": float(t["r2_std"]),
                    "mae_pct_mean": float(100.0 * mae_mean / max(y_range, 1e-12)),
                    "mae_pct_std": float(
                        100.0 * t.get("mae_std", 0.0) / max(y_range, 1e-12)
                    ),
                }
            measured[model_key][str(n_eff)] = entry

    series: dict[str, list[dict]] = {key: [] for key in MODELS}
    ns_measured = sorted({int(n) for m in measured.values() for n in m})
    plot_ns = sorted(set(ns) | (set(ns_measured) if ns_measured else set()))

    for n in plot_ns:
        n_eff = min(n, n_total)
        key = str(n_eff)
        is_measured = n <= n_total and n_eff >= n_splits and key in measured.get("random_forest", {})
        for model_key in MODELS:
            if is_measured and key in measured[model_key]:
                e = measured[model_key][key]
                cl_r2 = e["CL"]["r2_mean"]
                cd_r2 = e["CD"]["r2_mean"]
                series[model_key].append(
                    {
                        "n": n,
                        "measured": True,
                        "cv_r2_cl_mean": e["CL"]["r2_mean"],
                        "cv_r2_cl_std": e["CL"]["r2_std"],
                        "cv_r2_cd_mean": e["CD"]["r2_mean"],
                        "cv_r2_cd_std": e["CD"]["r2_std"],
                        "cv_r2_avg_mean": float((cl_r2 + cd_r2) / 2.0),
                        "cv_r2_avg_std": float(
                            np.sqrt(e["CL"]["r2_std"] ** 2 + e["CD"]["r2_std"] ** 2) / 2.0
                        ),
                        "cl_mae_pct_mean": e["CL"]["mae_pct_mean"],
                        "cl_mae_pct_std": e["CL"]["mae_pct_std"],
                    }
                )
            elif ns_measured:
                prev = [p for p in series[model_key] if p["measured"]]
                prev_ns = [int(p["n"]) for p in prev]
                prev_avg = [p["cv_r2_avg_mean"] for p in prev]
                prev_mae = [p["cl_mae_pct_mean"] for p in prev]
                series[model_key].append(
                    {
                        "n": n,
                        "measured": False,
                        "cv_r2_avg_mean": _project_value(prev_ns, prev_avg, n),
                        "cv_r2_avg_std": None,
                        "cl_mae_pct_mean": _project_value(prev_ns, prev_mae, n),
                        "cl_mae_pct_std": None,
                    }
                )

    return {
        "n_total_available": n_total,
        "measured_ns": ns_measured,
        "target_ns": ns,
        "cv_strategy": {"type": "RepeatedKFold", "n_splits": n_splits, "n_repeats": n_repeats, "seed": seed},
        "models": {MODELS[k]: series[k] for k in MODELS},
    }


def plot_from_json(data: dict, out_path: Path) -> Path:
    apply_paper_style()
    colors = {"Random Forest": C_RF, "Extra Trees": C_ET}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    fig.suptitle(
        "Surrogate Learning Curves",
        fontsize=13,
        fontweight="bold",
        color="#0D1B2A",
        y=1.03,
    )
    n_avail = data["n_total_available"]
    fig.text(
        0.5,
        0.99,
        f"Dataset: {n_avail} CFD cases — filled markers = measured, open = projected",
        ha="center",
        fontsize=8.5,
        color="#64748B",
    )

    ax = axes[0]
    panel_label(ax, "a")
    for model_name, points in data["models"].items():
        color = colors[model_name]
        ns = [p["n"] for p in points]
        r2m = [p["cv_r2_avg_mean"] for p in points]
        r2s = [p.get("cv_r2_avg_std") or 0.0 for p in points]
        measured_mask = [p["measured"] for p in points]

        ax.plot(ns, r2m, "-", color=color, linewidth=2.0, label=model_name, zorder=2)
        lo = [m - s for m, s in zip(r2m, r2s)]
        hi = [m + s for m, s in zip(r2m, r2s)]
        ax.fill_between(ns, lo, hi, color=color, alpha=0.12, zorder=1)

        for n, m, is_m in zip(ns, r2m, measured_mask):
            if is_m:
                ax.plot(n, m, "o", color=color, markersize=7, markerfacecolor=color, zorder=4)
            else:
                ax.plot(n, m, "o", color=color, markersize=7, markerfacecolor="white", markeredgewidth=1.5, zorder=4)

    ax.axhline(0.95, color=C_CERT_GATE, linestyle="--", linewidth=1.3, alpha=0.9, label="Certification gate ($R^2$=0.95)")
    ax.axhline(0.80, color=C_DESIGN_GATE, linestyle="--", linewidth=1.3, alpha=0.9, label="Design gate ($R^2$=0.80)")
    ax.set_xlabel("Training set size $N$")
    ax.set_ylabel("Mean CV $R^2$ (CL + CD)")
    ax.set_title("Cross-validated $R^2$", pad=8, color="#0D1B2A")
    ax.set_ylim(-0.05, 1.02)
    style_axes(ax)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    panel_label(ax, "b")
    for model_name, points in data["models"].items():
        color = colors[model_name]
        ns = [p["n"] for p in points]
        mae_m = [p["cl_mae_pct_mean"] for p in points]
        mae_s = [p.get("cl_mae_pct_std") or 0.0 for p in points]
        measured_mask = [p["measured"] for p in points]

        ax.plot(ns, mae_m, "-", color=color, linewidth=2.0, label=model_name, zorder=2)
        ax.fill_between(
            ns,
            [max(0.0, m - s) for m, s in zip(mae_m, mae_s)],
            [m + s for m, s in zip(mae_m, mae_s)],
            color=color,
            alpha=0.12,
            zorder=1,
        )
        for n, m, is_m in zip(ns, mae_m, measured_mask):
            if is_m:
                ax.plot(n, m, "o", color=color, markersize=7, markerfacecolor=color, zorder=4)
            else:
                ax.plot(n, m, "o", color=color, markersize=7, markerfacecolor="white", markeredgewidth=1.5, zorder=4)

    ax.axhline(5.0, color=C_CERT_GATE, linestyle="--", linewidth=1.3, alpha=0.9, label="Readiness gate (5% of range)")
    ax.set_xlabel("Training set size $N$")
    ax.set_ylabel("CL MAE (% of target range)")
    ax.set_title("CL error vs. dataset size", pad=8, color="#0D1B2A")
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=8)

    return save_figure(fig, out_path, dpi=280)


def main() -> None:
    p = argparse.ArgumentParser(description="Learning curve experiment and Fig. 2.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    p.add_argument("--output-fig", type=Path, default=DEFAULT_FIG)
    p.add_argument("--ns", type=int, nargs="+", default=TARGET_NS)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    x = df[BASE_COLS].to_numpy(dtype=float)
    x, _ = augment_design_inputs_v1(x)
    y = df[["CL", "CD"]].to_numpy(dtype=float)
    target_cols = ["CL", "CD"]

    if len(df) < max(args.ns):
        print(f"[WARN] Dataset has {len(df)} samples; N > {len(df)} will be projected.")

    result = run_experiment(x, y, target_cols, args.ns, args.cv_splits, args.n_repeats, args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[DONE] learning curve JSON -> {args.output_json}")

    if not args.no_plot:
        path = plot_from_json(result, args.output_fig)
        print(f"[DONE] {path}")


if __name__ == "__main__":
    main()
