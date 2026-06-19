#!/usr/bin/env python3
"""Fig. 1 — Pipeline architecture diagram (static, no data required)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figure_utils import C_BG, C_INK, apply_paper_style, save_figure

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
OUTPUT = FIG_DIR / "fig1_pipeline_architecture.png"

COLOR_A = "#2E6F9E"
COLOR_B = "#3A9E6F"
COLOR_C = "#C2553A"
COLOR_CORE = "#5B4B9A"
COLOR_UI = "#C9A227"
LANE_BG = {"A": "#EAF2FA", "B": "#E8F6EF", "C": "#FDEEEA"}


def _rounded_box(ax, xy, text, facecolor, textcolor="white", width=2.1, height=0.62, fontsize=11):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        linewidth=0, facecolor=facecolor, alpha=0.95, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=fontsize, color=textcolor,
        fontweight="semibold", zorder=4,
    )
    return x + width, y + height / 2


def _arrow(ax, start, end, color="#64748B", style="-|>", lw=1.5):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=11,
        linewidth=lw, color=color, zorder=2, shrinkA=2, shrinkB=2,
    ))


def plot_pipeline_architecture(out_path: Path = OUTPUT) -> Path:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.suptitle("Aero-ML Pipeline Architecture", fontsize=15, fontweight="bold", color=C_INK, y=0.98)

    # Swim lanes
    lanes = [
        ("A", "SU2 Design-Space", 6.2, 1.35, COLOR_A, LANE_BG["A"]),
        ("B", "SU2 AoA Sweep", 3.55, 1.2, COLOR_B, LANE_BG["B"]),
        ("C", "SolidWorks Export", 1.55, 1.1, COLOR_C, LANE_BG["C"]),
    ]
    for tag, title, y, h, color, bg in lanes:
        ax.add_patch(Rectangle((0.15, y - 0.18), 16.5, h + 0.36, facecolor=bg, edgecolor="none", zorder=0))
        ax.text(0.35, y + h - 0.08, f"Pipeline {tag}", fontsize=10, fontweight="bold", color=color, va="top")
        ax.text(0.35, y + h - 0.38, title, fontsize=8.5, color="#64748B", va="top")

    # Pipeline A
    a_steps = [
        "generate_design\n_space.py",
        "run_design\n_sweep.py",
        "build_dataset.py",
        "train_design\n_baseline.py",
        "optimize_design.py",
    ]
    y_a = 6.05
    x = 2.0
    prev = None
    for label in a_steps:
        end = _rounded_box(ax, (x, y_a), label, COLOR_A, width=2.05, height=0.72, fontsize=11)
        if prev:
            _arrow(ax, (prev[0] + 0.04, prev[1]), (x - 0.04, y_a + 0.36), COLOR_A)
        prev = end
        x += 2.35

    # Pipeline B
    b_steps = ["run_sweep.sh", "build_dataset.py", "train_baseline.py\n/ train_nn.py"]
    y_b = 3.45
    x = 2.0
    prev = None
    for label in b_steps:
        end = _rounded_box(ax, (x, y_b), label, COLOR_B, width=2.8, height=0.72, fontsize=11)
        if prev:
            _arrow(ax, (prev[0] + 0.04, prev[1]), (x - 0.04, y_b + 0.36), COLOR_B)
        prev = end
        x += 3.5

    # Pipeline C
    c_steps = ["SolidWorks\nExport", "train.py"]
    y_c = 1.55
    x = 2.0
    prev = None
    for label in c_steps:
        end = _rounded_box(ax, (x, y_c), label, COLOR_C, width=2.6, height=0.72, fontsize=11)
        if prev:
            _arrow(ax, (prev[0] + 0.04, prev[1]), (x - 0.04, y_c + 0.36), COLOR_C)
        prev = end
        x += 3.6

    # Shared ML core
    core_x, core_y, core_w, core_h = 5.8, 0.15, 4.6, 0.95
    ax.add_patch(FancyBboxPatch(
        (core_x, core_y), core_w, core_h,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        linewidth=0, facecolor=COLOR_CORE, alpha=0.96, zorder=3,
    ))
    ax.text(core_x + core_w / 2, core_y + core_h / 2,
            "Shared ML Core\nPyTorch  ·  scikit-learn  ·  evaluation",
            ha="center", va="center", fontsize=11, color="white", fontweight="bold", zorder=4)

    # CFD validation loop below ML core
    val_x, val_y = core_x + core_w * 0.28, core_y - 0.95
    _rounded_box(ax, (val_x, val_y), "validate_optimization.py", COLOR_A, width=3.2, height=0.72, fontsize=11)
    _arrow(ax, (core_x + core_w / 2, core_y), (val_x + 1.6, val_y + 0.72), COLOR_CORE, lw=1.8)

    # Merge arrows from lane ends down to core
    _arrow(ax, (12.8, y_a + 0.36), (core_x + core_w * 0.25, core_y + core_h), COLOR_A, lw=1.8)
    _arrow(ax, (10.3, y_b + 0.36), (core_x + core_w * 0.50, core_y + core_h), COLOR_B, lw=1.8)
    _arrow(ax, (8.2, y_c + 0.36), (core_x + core_w * 0.75, core_y + core_h), COLOR_C, lw=1.8)

    # Deployment interfaces
    ui_labels = ["FastAPI\nWeb UI", "Interactive\nCLI", "Tkinter\nPopup", "Batch\nJSON"]
    ui_y = 0.2
    ui_w = 1.95
    start_x = 11.2
    core_right = (core_x + core_w, core_y + core_h / 2)
    for i, label in enumerate(ui_labels):
        x = start_x + i * 1.35
        _rounded_box(ax, (x, ui_y), label, COLOR_UI, textcolor=C_INK, width=ui_w, height=0.85, fontsize=11)
        _arrow(ax, core_right, (x - 0.02, ui_y + 0.42), COLOR_UI, lw=1.4)
        core_right = (x + ui_w / 2, ui_y + 0.85)

    ax.text(11.0, 1.15, "Deployment interfaces", fontsize=9, fontweight="semibold", color="#64748B")

    return save_figure(fig, out_path, dpi=280)


def main() -> None:
    path = plot_pipeline_architecture()
    print(f"[DONE] {path}")


if __name__ == "__main__":
    main()
