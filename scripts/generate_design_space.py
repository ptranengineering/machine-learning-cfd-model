#!/usr/bin/env python3
"""
Generate multi-dimensional aerodynamic design-space samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "datasets" / "raw" / "design_space.csv"


def lhs(n_samples: int, n_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((n_samples, n_dim), dtype=float)
    for j in range(n_dim):
        bins = np.linspace(0.0, 1.0, n_samples + 1)
        pts = bins[:-1] + rng.uniform(0.0, 1.0 / n_samples, size=n_samples)
        rng.shuffle(pts)
        x[:, j] = pts
    return x


def main() -> None:
    p = argparse.ArgumentParser(description="Generate geometry + flow design-space samples.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n-samples", type=int, default=40)
    p.add_argument("--sampler", choices=["lhs", "random"], default="lhs")
    p.add_argument("--seed", type=int, default=42)
    # Optional bound overrides so you can narrow the space for SU2 convergence / surrogate quality.
    p.add_argument(
        "--geometry-thickness-range",
        nargs=2,
        type=float,
        default=[0.10, 0.16],
        metavar=("LO", "HI"),
        help="Thickness bounds (fraction of chord)",
    )
    p.add_argument(
        "--geometry-camber-range",
        nargs=2,
        type=float,
        default=[0.00, 0.04],
        metavar=("LO", "HI"),
        help="Camber magnitude bounds",
    )
    p.add_argument(
        "--geometry-camber-pos-range",
        nargs=2,
        type=float,
        default=[0.30, 0.50],
        metavar=("LO", "HI"),
        help="Position of maximum camber (fraction of chord)",
    )
    p.add_argument("--aoa-range", nargs=2, type=float, default=[0.0, 6.0], metavar=("LO", "HI"), help="Angle of attack (deg)")
    p.add_argument("--mach-range", nargs=2, type=float, default=[0.68, 0.80], metavar=("LO", "HI"))
    p.add_argument(
        "--reynolds-range",
        nargs=2,
        type=float,
        default=[3.0e6, 10.0e6],
        metavar=("LO", "HI"),
        help="Freestream Reynolds number",
    )
    args = p.parse_args()

    # Initial geometry representation:
    # thickness, camber, camber_position (simple parametric basis)
    # Conservative bounds to reduce mesh-pathology risk during geometry deformation.
    ranges = {
        "geometry_thickness": (args.geometry_thickness_range[0], args.geometry_thickness_range[1]),
        "geometry_camber": (args.geometry_camber_range[0], args.geometry_camber_range[1]),
        "geometry_camber_pos": (args.geometry_camber_pos_range[0], args.geometry_camber_pos_range[1]),
        "aoa": (args.aoa_range[0], args.aoa_range[1]),
        "mach": (args.mach_range[0], args.mach_range[1]),
        "reynolds": (args.reynolds_range[0], args.reynolds_range[1]),
    }
    columns = list(ranges.keys())

    n_dim = len(columns)
    if args.sampler == "lhs":
        u = lhs(args.n_samples, n_dim, args.seed)
    else:
        rng = np.random.default_rng(args.seed)
        u = rng.uniform(0.0, 1.0, size=(args.n_samples, n_dim))

    data = {}
    for i, col in enumerate(columns):
        lo, hi = ranges[col]
        data[col] = lo + (hi - lo) * u[:, i]

    df = pd.DataFrame(data)
    df.insert(0, "design_id", [f"design_{i+1:04d}" for i in range(len(df))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[DONE] wrote {len(df)} design samples to {args.output}")


if __name__ == "__main__":
    main()
