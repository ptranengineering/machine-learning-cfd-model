#!/usr/bin/env python3
"""Plot Euler vs. RANS coefficient comparison for ablation study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import C_INK, FIG_DIR, apply_paper_style, panel_label, save_figure, style_axes

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "results" / "euler_vs_rans_comparison.csv"
DEFAULT_OUT = FIG_DIR / "fig_euler_vs_rans.png"


def plot_comparison(df: pd.DataFrame, out_path: Path) -> Path:
    apply_paper_style()
    mach = df["mach"].to_numpy(dtype=float)
    cl_order = np.argsort(df["CL_rans"].fillna(0).to_numpy())
    mach_rank = np.empty_like(mach, dtype=float)
    mach_rank[cl_order] = np.linspace(0.0, 1.0, len(mach))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    fig.suptitle("Euler vs. RANS Ablation", fontsize=13, fontweight="bold", color=C_INK, y=1.02)

    panels = [
        ("CD_rans", "CD_euler", "RANS $C_D$", "Euler $C_D$", "a"),
        ("CL_rans", "CL_euler", "RANS $C_L$", "Euler $C_L$", "b"),
    ]
    for ax, (xcol, ycol, xlab, ylab, panel) in zip(axes, panels):
        x = df[xcol].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        sc = ax.scatter(x[mask], y[mask], c=mach_rank[mask], cmap="RdPu", s=48, alpha=0.88,
                        edgecolors="white", linewidths=0.5, vmin=0, vmax=1)
        lo = float(min(x[mask].min(), y[mask].min()))
        hi = float(max(x[mask].max(), y[mask].max()))
        pad = 0.06 * (hi - lo + 1e-9)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], color=C_INK, linestyle="--", linewidth=1.1, alpha=0.55)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_aspect("equal", adjustable="box")
        style_axes(ax)
        panel_label(ax, panel)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Mach (rank by CL)", fontsize=9)
    return save_figure(fig, out_path, dpi=220)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Euler vs. RANS ablation results.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Missing ablation data: {args.data}")

    df = pd.read_csv(args.data)
    path = plot_comparison(df, args.output)
    print(f"[DONE] {path}")


if __name__ == "__main__":
    main()
