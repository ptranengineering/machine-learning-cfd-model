#!/usr/bin/env python3
"""
Interactive surrogate optimization: enter design/flow ranges in the terminal (or paste JSON),
then search for a good design targeting CL and CD (lift/drag coefficients).

Run from repo root:
    python ml/interactive_optimize_design.py

Or from ml/:
    python interactive_optimize_design.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python ml/interactive_optimize_design.py` from repo root
_ML_DIR = Path(__file__).resolve().parent
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

import joblib
import numpy as np

from optimize_design import (
    DEFAULT_MODEL,
    RANGE_PARAM_NAMES,
    default_bounds,
    run_surrogate_optimization,
)


ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = ROOT / "results" / "interactive_design_optimization_result.json"

_LABELS = [
    "Geometry thickness (fraction of chord)",
    "Geometry camber (max camber, chord fraction)",
    "Geometry camber position (fraction chord to max camber)",
    "Angle of attack AoA (degrees)",
    "Mach number",
    "Reynolds number",
]


def _prompt_float_pair(default_lo: float, default_hi: float, label: str) -> tuple[float, float]:
    default_str = f"{default_lo:g} {default_hi:g}"
    raw = input(f"  {label}\n    min max [{default_str}]: ").strip()
    if not raw:
        return default_lo, default_hi
    parts = raw.replace(",", " ").split()
    if len(parts) != 2:
        print("    (!) Need two numbers: min max — using defaults.")
        return default_lo, default_hi
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        print("    (!) Invalid numbers — using defaults.")
        return default_lo, default_hi
    if hi <= lo:
        print("    (!) max must be > min — using defaults.")
        return default_lo, default_hi
    return lo, hi


def _parse_bounds_json_line(line: str) -> dict[str, list[float]]:
    data = json.loads(line)
    out: dict[str, list[float]] = {}
    for name in RANGE_PARAM_NAMES:
        pair = data.get(name)
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f'Missing or invalid "{name}": need [min, max]')
        lo, hi = float(pair[0]), float(pair[1])
        if hi <= lo:
            raise ValueError(f'"{name}": max must be greater than min')
        out[name] = [lo, hi]
    return out


def main() -> None:
    print(
        """
================================================================================
  Design surrogate optimizer (CL / CD — lift & drag coefficients)
================================================================================
  You will set a BOX in (thickness, camber, camber position, AoA, Mach, Re).
  The tool searches INSIDE that box using your trained surrogate.

  Goal options:
    (1) Maximize lift-to-drag ratio  CL/CD   [default]
    (2) Maximize lift                CL      (within drag cap)
    (3) Minimize drag                CD      (within lift floor)

  Constraints (applied to surrogate predictions during search):
    - Minimum CL (--style floor)
    - Maximum CD (--style ceiling)
================================================================================
"""
    )

    model_path = DEFAULT_MODEL
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1]).resolve()
    elif not model_path.exists():
        alt = ROOT / "results" / "models" / "design_rf_model.joblib"
        if alt.exists():
            model_path = alt

    mp = Path(model_path)
    if not mp.exists():
        print(f"Model not found: {mp}\nTrain first: python ml/train_design_baseline.py", file=sys.stderr)
        sys.exit(1)

    print(f"Using model: {mp}\n")

    d_lo, d_hi = default_bounds()

    print("How do you want to enter search ranges?")
    print("  [J] Paste ONE line of JSON (then Enter)")
    print("  [I] Interactive prompts (press Enter)")
    mode = input("Choice [I/j]: ").strip().lower() or "i"

    bounds_map: dict[str, list[float]] = {}
    if mode.startswith("j"):
        print("Paste JSON on ONE line, e.g.:")
        print(
            json.dumps(
                {
                    "geometry_thickness": [0.10, 0.15],
                    "geometry_camber": [0.0, 0.04],
                    "geometry_camber_pos": [0.30, 0.50],
                    "aoa": [0.0, 6.0],
                    "mach": [0.68, 0.80],
                    "reynolds": [3e6, 10e6],
                }
            )
        )
        line = input("JSON: ").strip()
        try:
            bounds_map = _parse_bounds_json_line(line)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\nEnter min max for each (space-separated); blank line = script default.")
        for i, (name, label) in enumerate(zip(RANGE_PARAM_NAMES, _LABELS, strict=True)):
            lo_i, hi_i = _prompt_float_pair(float(d_lo[i]), float(d_hi[i]), f"{label}  ({name})")
            bounds_map[name] = [lo_i, hi_i]

    lo = np.array([bounds_map[n][0] for n in RANGE_PARAM_NAMES], dtype=float)
    hi = np.array([bounds_map[n][1] for n in RANGE_PARAM_NAMES], dtype=float)

    print("\n--- Objective ---")
    print("  1 = maximize CL / CD")
    print("  2 = maximize CL")
    print("  3 = minimize CD")
    obj_in = input("Choice [1]: ").strip() or "1"
    objective = {"1": "max_cl_cd", "2": "max_cl", "3": "min_cd"}.get(obj_in, "max_cl_cd")

    mc = input("Minimum predicted CL allowed during search [0.7]: ").strip() or "0.7"
    xd = input("Maximum predicted CD allowed during search [0.2]: ").strip() or "0.2"
    min_cl, max_cd = float(mc), float(xd)

    it = input("BO iterations [20]: ").strip() or "20"
    init = input("Initial random samples [25]: ").strip() or "25"
    pool = input("Candidate pool per iter [500]: ").strip() or "500"
    sd = input("Random seed [42]: ").strip() or "42"

    pack = joblib.load(mp)
    result = run_surrogate_optimization(
        pack,
        lo,
        hi,
        objective=objective,
        min_cl=min_cl,
        max_cd=max_cd,
        iters=int(it),
        init_samples=int(init),
        candidate_pool=int(pool),
        seed=int(sd),
    )

    out_path = _DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("RESULT (surrogate predictions — validate in SU2 if decisions matter)")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
