"""
Shared feature engineering helpers for geometry+flow design surrogates.

The base inputs are ordered as:
geometry_param_1, geometry_param_2, geometry_param_3, AoA, Mach, Re
(which map to thickness, camber, camber position, angle of attack, Mach, Reynolds).
"""

from __future__ import annotations

import numpy as np

BASE_COLS = ["geometry_param_1", "geometry_param_2", "geometry_param_3", "AoA", "Mach", "Re"]

EXTRA_COLS_V1 = [
    "sin_AoA",
    "cos_AoA",
    "log10_Re",
    "AoA_x_Mach",
    "thick_x_AoA",
    "camber_x_camberPos",
    "Mach_sq",
]


def augment_design_inputs_v1(x_base: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Append lightweight nonlinear interactions / transforms used by ExtraTrees surrogates."""
    if x_base.ndim != 2 or x_base.shape[1] != len(BASE_COLS):
        raise ValueError(f"Expected shape (n, {len(BASE_COLS)}), got {getattr(x_base, 'shape', None)}")

    thickness = x_base[:, 0]
    camber = x_base[:, 1]
    cpp = x_base[:, 2]
    aoa = x_base[:, 3]
    mach = x_base[:, 4]
    re = x_base[:, 5]

    log10_re = np.log10(np.maximum(re, 1.0))
    aoa_rad = np.deg2rad(aoa)
    extras = np.column_stack(
        [
            np.sin(aoa_rad),
            np.cos(aoa_rad),
            log10_re,
            aoa * mach,
            thickness * aoa,
            camber * cpp,
            mach * mach,
        ]
    )
    cols = [*BASE_COLS, *EXTRA_COLS_V1]
    return np.concatenate([x_base, extras], axis=1), cols
