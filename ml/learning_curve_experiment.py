#!/usr/bin/env python3
"""
Fig. 2 — Surrogate learning curves (R² and MAE vs. training set size).

Requires datasets/processed/aero_design_dataset.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold

from design_feature_utils import BASE_COLS, augment_design_inputs_v1
from figure_utils import (
    C_ET,
    C_GATE,
    C_MLP,
    C_RF,
    FIG_DIR,
    apply_paper_style,
    panel_label,
    save_figure,
    style_axes,
    make_et_model,
    make_mlp_model,
    make_rf_model,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "datasets" / "processed" / "aero_design_dataset.csv"
OUTPUT = FIG_DIR / "fig2_learning_curves.png"

MODEL_BUILDERS = {
    "Random Forest": make_rf_model,
    "Extra Trees": make_et_model,
    "MLP": make_mlp_model,
}
MODEL_COLORS = {
    "Random Forest": C_RF,
    "Extra Trees": C_ET,
    "MLP": C_MLP,
}


def run_learning_curve(
    x: np.ndarray,
    y: np.ndarray,
    target_cols: list[str],
    ns: list[int],
    n_repeats: int,
    n_splits: int,
    seed: int,
) -> dict:
    n_total = len(x)
    rng = np.random.default_rng(seed)
    results: dict = {name: {t: {"r2": [], "mae_pct": []} for t in target_cols} for name in MODEL_BUILDERS}
    valid_ns: list[int] = []

    for n in ns:
        n_eff = min(n, n_total)
        if n_eff < 8:
            continue
        valid_ns.append(n)

        for model_name, builder in MODEL_BUILDERS.items():
            for target_idx, target in enumerate(target_cols):
                r2_folds: list[float] = []
                mae_pct_folds: list[float] = []

                for rep in range(n_repeats):
                    idx = rng.choice(n_total, size=n_eff, replace=False)
                    x_sub, y_sub = x[idx], y[idx]
                    if len(x_sub) < n_splits:
                        continue
                    y_range = float(np.max(y_sub[:, target_idx]) - np.min(y_sub[:, target_idx]))
                    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=seed + rep)
                    for train_idx, test_idx in rkf.split(x_sub):
                        model = builder(seed + rep)
                        model.fit(x_sub[train_idx], y_sub[train_idx])
                        pred = model.predict(x_sub[test_idx])
                        yt = y_sub[test_idx, target_idx]
                        yp = pred[:, target_idx]
                        r2_folds.append(float(r2_score(yt, yp)))
                        mae_pct_folds.append(
                            float(100.0 * mean_absolute_error(yt, yp) / max(y_range, 1e-12))
                        )

                if not r2_folds:
                    results[model_name][target]["r2"].append((np.nan, np.nan))
                    results[model_name][target]["mae_pct"].append((np.nan, np.nan))
                else:
                    results[model_name][target]["r2"].append((np.mean(r2_folds), np.std(r2_folds)))
                    results[model_name][target]["mae_pct"].append((np.mean(mae_pct_folds), np.std(mae_pct_folds)))

    return {"ns": valid_ns, "results": results}


def plot_learning_curves(curve_data: dict, out_path: Path, n_samples: int, cl_target: str = "CL") -> Path:
    apply_paper_style()
    ns = curve_data["ns"]
    results = curve_data["results"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    fig.suptitle("Surrogate Learning Curves", fontsize=13, fontweight="bold", color="#1F2A37", y=1.03)
    mlp_note = " (MLP omitted — needs $N \\geq 100$)" if n_samples < 100 else ""
    fig.text(
        0.5, 0.99,
        f"Dataset: {n_samples} CFD cases{mlp_note} — expand to 200–500+ for full learning curve",
        ha="center", fontsize=8.5, color="#64748B",
    )

    stable_models = ("Random Forest", "Extra Trees")
    scale_r2: list[float] = []
    scale_mae: list[float] = []

    ax = axes[0]
    panel_label(ax, "a")
    skip_mlp = n_samples < 100
    for model_name, color in MODEL_COLORS.items():
        if model_name == "MLP" and skip_mlp:
            continue
        r2_means, r2_stds = [], []
        for i in range(len(ns)):
            cl_m, cl_s = results[model_name]["CL"]["r2"][i]
            cd_m, cd_s = results[model_name]["CD"]["r2"][i]
            r2_means.append((cl_m + cd_m) / 2.0)
            r2_stds.append(np.sqrt((cl_s**2 + cd_s**2) / 2.0))
        r2_means = np.array(r2_means)
        r2_stds = np.array(r2_stds)
        if model_name in stable_models:
            scale_r2.extend((r2_means - r2_stds).tolist())
            scale_r2.extend((r2_means + r2_stds).tolist())
        lo_band = r2_means - r2_stds
        hi_band = r2_means + r2_stds
        ax.plot(ns, r2_means, "o-", color=color, label=model_name, linewidth=2.2, markersize=6,
                markerfacecolor="white", markeredgewidth=1.3)
        ax.fill_between(ns, lo_band, hi_band, color=color, alpha=0.12)
    ax.axhline(0.95, color=C_GATE, linestyle="--", linewidth=1.3, alpha=0.85, label="Readiness gate ($R^2$ = 0.95)")
    ax.set_xlabel("Training set size $N$")
    ax.set_ylabel("Mean CV $R^2$ (CL + CD)")
    ax.set_title("Cross-validated $R^2$", pad=8)
    r2_lo = min(-0.3, float(np.nanmin(scale_r2)) - 0.08) if scale_r2 else -0.3
    ax.set_ylim(r2_lo, 1.02)
    style_axes(ax)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    panel_label(ax, "b")
    for model_name, color in MODEL_COLORS.items():
        if model_name == "MLP" and skip_mlp:
            continue
        mae_means, mae_stds = [], []
        for i in range(len(ns)):
            m, s = results[model_name][cl_target]["mae_pct"][i]
            mae_means.append(m)
            mae_stds.append(s)
        mae_means = np.array(mae_means)
        mae_stds = np.array(mae_stds)
        if model_name in stable_models:
            scale_mae.extend((mae_means - mae_stds).tolist())
            scale_mae.extend((mae_means + mae_stds).tolist())
        ax.plot(ns, mae_means, "o-", color=color, label=model_name, linewidth=2.2, markersize=6,
                markerfacecolor="white", markeredgewidth=1.3)
        ax.fill_between(ns, np.maximum(mae_means - mae_stds, 0), mae_means + mae_stds, color=color, alpha=0.12)
    ax.axhline(5.0, color=C_GATE, linestyle="--", linewidth=1.3, alpha=0.85, label="Readiness gate (5% of range)")
    ax.set_xlabel("Training set size $N$")
    ax.set_ylabel(f"{cl_target} MAE (% of target range)")
    ax.set_title(f"{cl_target} error vs. dataset size", pad=8)
    mae_hi = min(30.0, max(20.0, float(np.nanmax(scale_mae)) + 2.0)) if scale_mae else 20.0
    ax.set_ylim(0, mae_hi)
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=8)

    return save_figure(fig, out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Fig. 2 learning curves.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--ns", type=int, nargs="+", default=[50, 100, 150, 200, 300, 400, 500])
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    x = df[BASE_COLS].to_numpy(dtype=float)
    x, _ = augment_design_inputs_v1(x)
    y = df[["CL", "CD"]].to_numpy(dtype=float)
    target_cols = ["CL", "CD"]

    n_avail = len(df)
    ns_eff = sorted({min(n, n_avail) for n in args.ns if min(n, n_avail) >= 8})
    if n_avail < max(args.ns):
        print(f"[WARN] Dataset has {n_avail} samples; capping N values to <= {n_avail}.")

    curve = run_learning_curve(x, y, target_cols, ns_eff, args.n_repeats, args.cv_splits, args.seed)
    path = plot_learning_curves(curve, args.output, n_avail)
    print(f"[DONE] {path}")


if __name__ == "__main__":
    main()
