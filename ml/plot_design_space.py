#!/usr/bin/env python3
"""
Fig. 6 — Design space heatmap of CL/CD across AoA and thickness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from figure_utils import C_GATE, C_INK, FIG_DIR, apply_paper_style, save_figure, style_axes
from optimize_design import augment_if_needed

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "results" / "models" / "design_rf_model.joblib"
DEFAULT_DATA = ROOT / "datasets" / "processed" / "aero_design_dataset.csv"
DEFAULT_OPT = ROOT / "results" / "design_optimization_result.json"
OUTPUT = FIG_DIR / "fig6_design_heatmap.png"


def build_input_grid(aoa_grid: np.ndarray, thick_grid: np.ndarray, fixed: dict[str, float]) -> np.ndarray:
    rows = []
    for t in thick_grid:
        for a in aoa_grid:
            rows.append([
                t, fixed["geometry_camber"], fixed["geometry_camber_pos"],
                a, fixed["mach"], fixed["reynolds"],
            ])
    return np.array(rows, dtype=float)


def filter_training_overlay(df: pd.DataFrame, fixed: dict[str, float], tol: float = 0.10) -> pd.DataFrame:
    def near(col: str, val: float) -> pd.Series:
        if col not in df.columns:
            return pd.Series(True, index=df.index)
        return (df[col] - val).abs() <= tol * max(abs(val), 1e-9)

    mask = (
        near("geometry_param_2", fixed["geometry_camber"])
        & near("geometry_param_3", fixed["geometry_camber_pos"])
        & near("Mach", fixed["mach"])
        & near("Re", fixed["reynolds"])
    )
    return df.loc[mask]


def plot_design_heatmap(
    pack: dict,
    df: pd.DataFrame,
    fixed: dict[str, float],
    optimum: dict[str, float] | None,
    out_path: Path,
    aoa_range: tuple[float, float] = (-2.0, 14.0),
    thick_range: tuple[float, float] = (0.08, 0.18),
    n_grid: int = 150,
    contour_levels: list[float] | None = None,
) -> Path:
    if contour_levels is None:
        contour_levels = [10, 15, 20, 25]

    apply_paper_style()
    model = pack["model"]
    augment_version = int(pack.get("augment_version", 0))

    aoa = np.linspace(aoa_range[0], aoa_range[1], n_grid)
    thick = np.linspace(thick_range[0], thick_range[1], n_grid)
    x_base = build_input_grid(aoa, thick, fixed)
    x_aug = augment_if_needed(x_base, augment_version)
    pred = model.predict(x_aug)
    cl = pred[:, 0].reshape(len(thick), len(aoa))
    cd = np.maximum(pred[:, 1].reshape(len(thick), len(aoa)), 1e-12)
    cl_cd = cl / cd
    cl_cd = np.clip(cl_cd, 0, np.nanpercentile(cl_cd, 99))

    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    fig.suptitle("Surrogate CL/CD Landscape", fontsize=13, fontweight="bold", color=C_INK, y=1.02)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "aero_perf", ["#B83232", "#E8A838", "#F5F0C4", "#7BC67E", "#1F7A4D"], N=256,
    )
    vmin, vmax = float(np.nanpercentile(cl_cd, 2)), float(np.nanpercentile(cl_cd, 98))
    cf = ax.contourf(aoa, thick, cl_cd, levels=60, cmap=cmap, vmin=vmin, vmax=vmax, extend="both")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Predicted CL/CD", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    cs = ax.contour(aoa, thick, cl_cd, levels=contour_levels, colors="white", linewidths=0.9, alpha=0.75)
    ax.clabel(cs, inline=True, fontsize=7.5, fmt="%.0f", colors=C_INK)

    overlay = filter_training_overlay(df, fixed)
    if not overlay.empty:
        ax.scatter(
            overlay["AoA"], overlay["geometry_param_1"],
            s=22, c=C_INK, alpha=0.45, edgecolors="white", linewidths=0.4,
            label=f"Training data ({len(overlay)} pts)", zorder=5,
        )

    if optimum is not None:
        ax.scatter(
            [optimum["aoa"]], [optimum["geometry_thickness"]],
            marker="*", s=280, c=C_GATE, edgecolors="white", linewidths=1.0,
            label="Bayesian optimum", zorder=6,
        )

    ax.set_xlabel("Angle of attack (deg)")
    ax.set_ylabel("Thickness-to-chord ratio")
    style_axes(ax, grid=False)

    note = (
        f"Fixed parameters\n"
        f"camber = {fixed['geometry_camber']:.3f}\n"
        f"pos = {fixed['geometry_camber_pos']:.2f}\n"
        f"Mach = {fixed['mach']:.3f}\n"
        f"Re = {fixed['reynolds']:.2e}"
    )
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7.5, color="#64748B", va="bottom",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#E2E8F0", alpha=0.92))
    ax.legend(loc="upper right", framealpha=0.92, fontsize=8)

    return save_figure(fig, out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Fig. 6 design-space heatmap.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--opt-result", type=Path, default=DEFAULT_OPT)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()

    pack = joblib.load(args.model)
    df = pd.read_csv(args.data)

    fixed = {
        "geometry_camber": 0.02,
        "geometry_camber_pos": 0.40,
        "mach": 0.75,
        "reynolds": 5.0e6,
    }
    optimum = None
    if args.opt_result.exists():
        result = json.loads(args.opt_result.read_text(encoding="utf-8"))
        bgf = result.get("best_geometry_flow", {})
        fixed = {
            "geometry_camber": bgf.get("geometry_camber", fixed["geometry_camber"]),
            "geometry_camber_pos": bgf.get("geometry_camber_pos", fixed["geometry_camber_pos"]),
            "mach": bgf.get("mach", fixed["mach"]),
            "reynolds": bgf.get("reynolds", fixed["reynolds"]),
        }
        optimum = bgf

    path = plot_design_heatmap(pack, df, fixed, optimum, args.output)
    print(f"[DONE] {path}")


if __name__ == "__main__":
    main()
