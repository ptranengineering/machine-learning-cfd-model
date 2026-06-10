"""Shared plotting helpers and publication style for paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"

# Palette — muted, print-friendly
C_RF = "#2E6F9E"
C_ET = "#3A9E6F"
C_MLP = "#C2553A"
C_GATE = "#B83232"
C_ACCENT = "#D4A017"
C_MUTED = "#9AA3AD"
C_INK = "#1F2A37"
C_BG = "#F7F9FC"

FEATURE_LABELS = {
    "geometry_param_1": "Thickness",
    "geometry_param_2": "Camber",
    "geometry_param_3": "Camber position",
    "AoA": "Angle of attack",
    "Mach": "Mach number",
    "Re": "Reynolds number",
    "sin_AoA": "sin(AoA)",
    "cos_AoA": "cos(AoA)",
    "log10_Re": "log₁₀(Re)",
    "AoA_x_Mach": "AoA × Mach",
    "thick_x_AoA": "Thickness × AoA",
    "camber_x_camberPos": "Camber × position",
    "Mach_sq": "Mach²",
}


def apply_paper_style() -> None:
    """Global matplotlib rc for clean, publication-ready figures."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": C_INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.titlepad": 10,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#64748B",
            "ytick.color": "#64748B",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
            "legend.fontsize": 8.5,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#E2E8F0",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def style_axes(ax, *, grid: bool = True) -> None:
    ax.tick_params(length=4, width=0.8, colors="#64748B")
    if grid:
        ax.grid(True, axis="both", color="#E8EDF3", linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12, 1.06, label, transform=ax.transAxes,
        fontsize=12, fontweight="bold", color=C_INK, va="top", ha="left",
    )


def humanize_features(names: list[str]) -> list[str]:
    return [FEATURE_LABELS.get(n, n) for n in names]


def save_figure(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    return out_path


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-12)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def mae_pct_of_range(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_range = float(np.max(y_true) - np.min(y_true))
    return float(100.0 * mean_absolute_error(y_true, y_pred) / max(y_range, 1e-12))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def plot_parity_plots(
    x_test: np.ndarray,
    y_test: np.ndarray,
    model,
    feature_cols: list[str],
    target_cols: list[str],
    out_path: Path,
    model_label: str = "RF + engineered features",
) -> Path:
    """Fig. 3 — predicted vs. actual scatter plots for CL and CD."""
    apply_paper_style()
    pred = model.predict(x_test)
    mach_idx = feature_cols.index("Mach") if "Mach" in feature_cols else 4
    mach = x_test[:, mach_idx]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    fig.suptitle("Surrogate Validation: Predicted vs. CFD", fontsize=13, fontweight="bold", color=C_INK, y=1.02)
    fig.text(0.5, 0.97, model_label, ha="center", fontsize=9.5, color="#64748B")

    for ax, i, target, panel in zip(axes, range(len(target_cols)), target_cols, ("a", "b")):
        yt = y_test[:, i]
        yp = pred[:, i]
        sc = ax.scatter(
            yt, yp, c=mach, cmap="plasma", s=42, alpha=0.82,
            edgecolors="white", linewidths=0.5, zorder=3,
        )
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        pad = 0.06 * (hi - lo + 1e-9)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], color=C_INK, linestyle="--", linewidth=1.1, alpha=0.55, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"CFD {target}")
        ax.set_ylabel(f"Surrogate {target}")
        ax.set_title(f"{target} coefficient", pad=8)
        ax.set_aspect("equal", adjustable="box")
        style_axes(ax)
        panel_label(ax, panel)

        r2 = r2_score(yt, yp)
        stats = f"$R^2$ = {r2:.3f}\nRMSE = {rmse(yt, yp):.4f}\nMAE = {mae_pct_of_range(yt, yp):.1f}% of range"
        ax.text(
            0.97, 0.05, stats, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#E2E8F0", alpha=0.95),
        )
        if i == 1:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Mach number", fontsize=9)
            cbar.ax.tick_params(labelsize=8)

    return save_figure(fig, out_path)


def plot_feature_importance(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    target_cols: list[str],
    out_path: Path,
    seed: int = 42,
    n_repeats: int = 10,
) -> Path:
    """Fig. 4 — permutation feature importance for CL and CD."""
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), constrained_layout=True)
    fig.suptitle("Permutation Feature Importance", fontsize=13, fontweight="bold", color=C_INK, y=1.02)

    estimators = model.estimators_ if hasattr(model, "estimators_") else [model, model]

    for ax, i, target, panel in zip(axes, range(len(target_cols)), target_cols, ("a", "b")):
        est = estimators[i]
        result = permutation_importance(
            est, x_test, y_test[:, i], n_repeats=n_repeats, random_state=seed, n_jobs=-1,
        )
        order = np.argsort(result.importances_mean)
        names = humanize_features([feature_cols[j] for j in order])
        means = result.importances_mean[order]
        stds = result.importances_std[order]
        bar_colors = [C_RF if m >= 0 else "#94A3B8" for m in means]
        ax.barh(names, means, xerr=stds, color=bar_colors, alpha=0.9, capsize=2.5,
                error_kw={"elinewidth": 0.9, "ecolor": "#64748B", "alpha": 0.7})
        ax.axvline(0.0, color=C_INK, linewidth=0.8, alpha=0.35)
        ax.set_xlabel("Permutation importance (Δ score)")
        ax.set_title(f"{target} prediction", pad=8)
        style_axes(ax, grid=True)
        panel_label(ax, panel)

    return save_figure(fig, out_path)


def plot_bo_convergence(
    histories: list[np.ndarray],
    optimum_cl_cd: float,
    out_path: Path,
    cfd_verified_cl_cd: float | None = None,
) -> Path:
    """Fig. 5 — Bayesian optimization incumbent convergence across seeds."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    fig.suptitle("Bayesian Optimization Convergence", fontsize=13, fontweight="bold", color=C_INK, y=1.02)

    max_len = max(len(h) for h in histories)
    padded = []
    for h in histories:
        arr = np.maximum.accumulate(np.asarray(h, dtype=float))
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), mode="edge")
        padded.append(arr)

    stack = np.vstack(padded)
    iters = np.arange(stack.shape[1])
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)

    for h in padded:
        ax.plot(iters, h, color="#D1D9E0", linewidth=0.9, alpha=0.55, zorder=1)
    ax.fill_between(iters, mean - std, mean + std, color=C_RF, alpha=0.18, zorder=2, label="±1 std")
    ax.plot(iters, mean, color=C_RF, linewidth=2.8, marker="o", markersize=4.5,
            markerfacecolor="white", markeredgewidth=1.2, zorder=4, label="Mean incumbent")
    ax.scatter(
        [iters[-1]], [optimum_cl_cd], marker="*", s=220, color=C_GATE,
        edgecolors="white", linewidths=1.0, zorder=5,
        label=f"Surrogate optimum ({optimum_cl_cd:.1f})",
    )
    if cfd_verified_cl_cd is not None:
        ax.axhline(
            cfd_verified_cl_cd, color=C_GATE, linestyle="--", linewidth=1.4, alpha=0.8,
            label=f"CFD-verified ({cfd_verified_cl_cd:.1f})",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best CL/CD (incumbent)")
    ymin = max(0.0, float(np.min(stack) - 1.0))
    ymax = float(np.max(stack) + 1.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.5, iters[-1] + 0.5)
    style_axes(ax)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.text(0.5, 0.01, "5 independent seeds; shaded band shows ±1 standard deviation",
             ha="center", fontsize=8.5, color="#64748B")

    return save_figure(fig, out_path)


def make_rf_model(seed: int) -> MultiOutputRegressor:
    return MultiOutputRegressor(RandomForestRegressor(n_estimators=500, random_state=seed))


def make_et_model(seed: int) -> MultiOutputRegressor:
    return MultiOutputRegressor(ExtraTreesRegressor(n_estimators=700, random_state=seed))


def make_mlp_model(seed: int) -> MultiOutputRegressor:
    return MultiOutputRegressor(
        MLPRegressor(
            hidden_layer_sizes=(96, 96, 48),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
    )
