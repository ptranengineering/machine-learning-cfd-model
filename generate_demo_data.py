"""
Write a synthetic CFD-like CSV for smoke-testing the training pipeline.

This is NOT a substitute for Solidworks CFD results — only checks that the code runs.

Usage:
    python generate_demo_data.py
    python train.py --data data/raw/cfd_demo_data.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

from config import INPUT_PARAMS, OUTPUT_PARAMS, PARAM_RANGES, RAW_DATA_DIR, RANDOM_STATE


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / "cfd_demo_data.csv"

    n_samples = 220
    names = INPUT_PARAMS
    lows = np.array([PARAM_RANGES[k][0] for k in names])
    highs = np.array([PARAM_RANGES[k][1] for k in names])

    sampler = qmc.LatinHypercube(d=len(names), seed=RANDOM_STATE)
    unit = sampler.random(n_samples)
    X = qmc.scale(unit, lows, highs)

    thickness = X[:, 0]
    camber = X[:, 1]
    position = X[:, 2]
    aoa = X[:, 3]
    reynolds = X[:, 4]

    rng = np.random.default_rng(RANDOM_STATE)
    aoa_rad = np.deg2rad(aoa)
    log_re = np.log10(np.clip(reynolds, 1e5, 1e7))

    # Smooth surrogate-ish mappings + noise (units loosely inspired by thin-airfoil scaling)
    Cl = (
        2.0 * np.pi * aoa_rad * (0.55 + 2.0 * camber)
        + 0.15 * (position - 0.3)
        + 0.02 * (log_re - 6.0)
        + 0.04 * rng.standard_normal(n_samples)
    )
    Cd = (
        0.009
        + 0.25 * Cl**2
        + 0.03 * thickness
        + 0.015 * np.abs(aoa_rad)
        + 5e-4 * np.maximum(log_re - 6.0, 0.0)
        + 0.003 * rng.standard_normal(n_samples)
    )
    Cd = np.clip(Cd, 1e-4, None)
    Cm = -0.22 * Cl + 0.03 * camber + 0.01 * rng.standard_normal(n_samples)

    df = pd.DataFrame(
        np.column_stack([thickness, camber, position, aoa, reynolds, Cd, Cl, Cm]),
        columns=INPUT_PARAMS + OUTPUT_PARAMS,
    )
    df.to_csv(out_path, index=False)

    print(f"Wrote {n_samples} synthetic samples to {out_path}")
    print("Train with:")
    print(f"  python train.py --data {out_path}")


if __name__ == "__main__":
    main()
