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
C_CERT_GATE = "#B83232"
C_DESIGN_GATE = "#E07B39"
C_ACCENT = "#D4A017"
C_MUTED = "#9AA3AD"
C_INK = "#0D1B2A"
C_BG = "#F8FAFC"
C_TEAL = "#1B6CA8"
C_CORAL = "#D45F3C"

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
            "axes.facecolor": C_BG,
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": C_INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.titlepad": 10,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#64748B",
            "ytick.color": "#64748B",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "serif",
            "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
            "legend.fontsize": 8.5,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#E2E8F0",
            "figure.dpi": 120,
            "savefig.dpi": 280,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def style_axes(ax, *, grid: bool = True) -> None:
    ax.tick_params(length=4, width=0.8, colors="#64748B")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis="both", color="#E2E8F0", linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, out_path: Path, dpi: int = 280) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12, 1.06, label, transform=ax.transAxes,
        fontsize=12, fontweight="bold", color=C_INK, va="top", ha="left",
    )


def humanize_features(names: list[str]) -> list[str]:
    return [FEATURE_LABELS.get(n, n) for n in names]


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
    cl_idx = target_cols.index("CL") if "CL" in target_cols else 0
    cl_order = np.argsort(y_test[:, cl_idx])
    mach_rank = np.empty_like(mach, dtype=float)
    mach_rank[cl_order] = np.linspace(0.0, 1.0, len(mach))

    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.055], wspace=0.28)
    ax_cl = fig.add_subplot(gs[0, 0])
    ax_cd = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    axes = [ax_cl, ax_cd]

    fig.suptitle("Surrogate Validation: Predicted vs. CFD", fontsize=13, fontweight="bold", color=C_INK, y=1.02)
    fig.text(0.5, 0.97, model_label, ha="center", fontsize=9.5, color="#64748B")

    scatter_ref = None
    for ax, i, target, panel in zip(axes, range(len(target_cols)), target_cols, ("a", "b")):
        yt = y_test[:, i]
        yp = pred[:, i]
        scatter_ref = ax.scatter(
            yt, yp, c=mach_rank, cmap="RdPu", s=42, alpha=0.85,
            edgecolors="white", linewidths=0.5, zorder=3, vmin=0, vmax=1,
        )
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        pad = 0.06 * (hi - lo + 1e-9)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], color=C_INK, linestyle="--", linewidth=1.1, alpha=0.55, zorder=2)
        band = 0.10 * (hi - lo)
        ax.fill_between([lo, hi], [lo - band, hi - band], [lo + band, hi + band],
                        color="#E2E8F0", alpha=0.45, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"CFD {target}")
        ax.set_ylabel(f"Surrogate {target}")
        ax.set_title(f"{target} coefficient", pad=8, color=C_INK)
        ax.set_aspect("equal", adjustable="box")
        style_axes(ax)
        panel_label(ax, panel)

        r2 = r2_score(yt, yp)
        stats = f"$R^2$ = {r2:.3f}\nRMSE = {rmse(yt, yp):.4f}\nMAE = {mae_pct_of_range(yt, yp):.1f}% of range"
        ax.text(
            0.03, 0.97, stats, transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#E2E8F0", alpha=0.95),
        )

    if scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, cax=cax)
        cbar.set_label("Mach (rank by CL)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.93)
    return save_figure(fig, out_path, dpi=220)


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
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)
    fig.suptitle("Permutation Feature Importance", fontsize=13, fontweight="bold", color=C_INK, y=1.02)

    estimators = model.estimators_ if hasattr(model, "estimators_") else [model, model]

    for ax, i, target, panel in zip(axes, range(len(target_cols)), target_cols, ("a", "b")):
        est = estimators[i]
        result = permutation_importance(
            est, x_test, y_test[:, i], n_repeats=n_repeats, random_state=seed, n_jobs=-1,
        )
        order = np.argsort(result.importances_mean)[::-1]
        names = humanize_features([feature_cols[j] for j in order])
        means = result.importances_mean[order]
        stds = result.importances_std[order]
        bar_colors = [C_TEAL if m >= 0 else C_CORAL for m in means]
        bars = ax.barh(names, means, xerr=stds, color=bar_colors, alpha=0.92, capsize=2.5,
                       error_kw={"elinewidth": 0.9, "ecolor": "#64748B", "alpha": 0.7})
        ax.axvline(0.0, color=C_INK, linewidth=0.8, alpha=0.35)
        for bar, val in zip(bars, means):
            ax.text(
                val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=7.5,
                color=C_INK,
            )
        ax.set_xlabel("Permutation importance (Δ score)")
        ax.set_title(f"{target} prediction", pad=8, color=C_INK)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        style_axes(ax, grid=True)
        panel_label(ax, panel)

    return save_figure(fig, out_path, dpi=220)


def plot_bo_convergence(
    histories: list[np.ndarray],
    optimum_cl_cd: float,
    out_path: Path,
    cfd_verified_cl_cd: float | None = None,
    within_training_hull: bool | None = None,
) -> Path:
    """Fig. 5 — Bayesian optimization incumbent convergence across seeds."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
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

    ax.axvspan(-0.5, 7.0, color="#DBEAFE", alpha=0.35, zorder=0, label="Exploration (0–7)")
    ax.axvspan(7.0, iters[-1] + 0.5, color="#D1FAE5", alpha=0.35, zorder=0, label="Exploitation (7–20)")

    for h in padded:
        ax.plot(iters, h, color="#93C5FD", linewidth=0.9, alpha=0.65, zorder=1)
    ax.fill_between(iters, mean - std, mean + std, color=C_TEAL, alpha=0.18, zorder=2, label="±1 std")
    ax.plot(iters, mean, color=C_TEAL, linewidth=2.8, marker="o", markersize=4.5,
            markerfacecolor="white", markeredgewidth=1.2, zorder=4, label="Mean incumbent")
    ax.scatter(
        [iters[-1]], [optimum_cl_cd], marker="*", s=220, color=C_GATE,
        edgecolors="white", linewidths=1.0, zorder=5,
        label=f"Optimum CL/CD = {optimum_cl_cd:.1f}",
    )
    ax.annotate(
        f"CL/CD = {optimum_cl_cd:.1f}",
        xy=(iters[-1], optimum_cl_cd),
        xytext=(iters[-1] - 4.5, optimum_cl_cd + 0.08 * max(optimum_cl_cd, 1.0)),
        fontsize=9,
        color=C_GATE,
        arrowprops=dict(arrowstyle="->", color=C_GATE, lw=1.0),
    )
    if cfd_verified_cl_cd is not None:
        ax.axhline(
            cfd_verified_cl_cd, color=C_GATE, linestyle="--", linewidth=1.4, alpha=0.8,
            label=f"CFD-verified ({cfd_verified_cl_cd:.1f})",
        )

    if within_training_hull is True:
        ax.text(
            0.02, 0.98, "Optimum within training hull",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#166534",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#DCFCE7", edgecolor="#86EFAC", alpha=0.95),
        )
    elif within_training_hull is False:
        ax.text(
            0.02, 0.98, "Warning: optimum outside training hull",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#991B1B",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEE2E2", edgecolor="#FCA5A5", alpha=0.95),
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best CL/CD (incumbent)")
    ymin = max(0.0, float(np.min(stack) - 1.0))
    ymax = float(np.max(stack) + 1.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.5, iters[-1] + 0.5)
    style_axes(ax)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
    fig.text(0.5, 0.01, "5 independent seeds; shaded band shows ±1 standard deviation",
             ha="center", fontsize=8.5, color="#64748B")

    return save_figure(fig, out_path, dpi=220)


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
