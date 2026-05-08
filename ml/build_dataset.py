#!/usr/bin/env python3
"""
Build a training-ready dataset by merging SU2 sweep outputs.

Expected raw layout:
  datasets/raw/su2_sweep_YYYYmmdd_HHMMSS/cases/AoA_<value>/
    - forces_breakdown_AoA_<value>.dat
    - history_AoA_<value>.csv (optional)
    - surface_flow_AoA_<value>.csv (optional)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_ROOT = ROOT_DIR / "datasets" / "raw"
DEFAULT_OUTPUT = ROOT_DIR / "datasets" / "processed" / "cfd_training_data.csv"
DEFAULT_CFG = ROOT_DIR / "su2_cases" / "inv_NACA0012.cfg"


def parse_cfg_value(cfg_path: Path, key: str) -> Optional[float]:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^\s%]+)")
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
    return None


def extract_coeff_from_forces(forces_path: Path, coeff: str) -> Optional[float]:
    # Matches lines like:
    # Total CL:       1.490346 | Pressure ...
    pattern = re.compile(rf"^Total\s+{re.escape(coeff)}\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")
    with forces_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                return float(match.group(1))
    return None


def list_sweep_dirs(raw_root: Path) -> list[Path]:
    return sorted(
        [p for p in raw_root.glob("su2_sweep_*") if p.is_dir()],
        key=lambda p: p.name,
    )


def parse_aoa_from_case_dir(case_dir: Path) -> Optional[float]:
    # Accept AoA_2 or AoA_2.5
    match = re.match(r"^AoA_([+-]?\d*\.?\d+)$", case_dir.name)
    if not match:
        return None
    return float(match.group(1))


def build_rows(
    sweep_dirs: Iterable[Path],
    default_mach: Optional[float],
    default_reynolds: Optional[float],
    geometry_id: str,
) -> list[dict]:
    rows: list[dict] = []

    for sweep_dir in sweep_dirs:
        run_id = sweep_dir.name.replace("su2_sweep_", "")
        cases_dir = sweep_dir / "cases"
        if not cases_dir.exists():
            continue

        for case_dir in sorted(p for p in cases_dir.iterdir() if p.is_dir()):
            aoa = parse_aoa_from_case_dir(case_dir)
            if aoa is None:
                continue

            candidate_forces = sorted(case_dir.glob("forces_breakdown*.dat"))
            if not candidate_forces:
                continue
            forces_file = candidate_forces[0]

            cl = extract_coeff_from_forces(forces_file, "CL")
            cd = extract_coeff_from_forces(forces_file, "CD")
            if cl is None or cd is None:
                continue

            ld_ratio = cl / cd if cd != 0 else None
            rows.append(
                {
                    "run_id": run_id,
                    "case_id": case_dir.name,
                    "geometry_id": geometry_id,
                    "aoa": aoa,
                    "mach": default_mach,
                    "reynolds": default_reynolds,
                    "cl": cl,
                    "cd": cd,
                    "ld_ratio": ld_ratio,
                    "forces_file": str(forces_file.relative_to(ROOT_DIR)),
                    "case_dir": str(case_dir.relative_to(ROOT_DIR)),
                }
            )
    return rows


def write_dataset(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "case_id",
        "geometry_id",
        "aoa",
        "mach",
        "reynolds",
        "cl",
        "cd",
        "ld_ratio",
        "forces_file",
        "case_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge SU2 sweep outputs into ML-ready CSV.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--geometry-id", type=str, default="NACA0012")
    parser.add_argument("--mach", type=float, default=None, help="Override Mach value.")
    parser.add_argument("--reynolds", type=float, default=None, help="Override Reynolds value.")
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Use only latest su2_sweep_* directory instead of all runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sweep_dirs = list_sweep_dirs(args.raw_root)
    if not sweep_dirs:
        raise FileNotFoundError(f"No su2_sweep_* directories found in: {args.raw_root}")

    if args.latest_only:
        sweep_dirs = [sweep_dirs[-1]]

    cfg_mach = parse_cfg_value(args.cfg, "MACH_NUMBER")
    cfg_re = parse_cfg_value(args.cfg, "REYNOLDS_NUMBER")
    mach = args.mach if args.mach is not None else cfg_mach
    reynolds = args.reynolds if args.reynolds is not None else cfg_re

    rows = build_rows(
        sweep_dirs=sweep_dirs,
        default_mach=mach,
        default_reynolds=reynolds,
        geometry_id=args.geometry_id,
    )
    if not rows:
        raise RuntimeError("No valid cases with parseable CL/CD were found.")

    # Keep data deterministic for reproducibility.
    rows = sorted(rows, key=lambda r: (r["run_id"], r["aoa"]))
    write_dataset(rows, args.output)

    print(f"[DONE] rows={len(rows)}")
    print(f"[DONE] output={args.output}")
    print(f"[INFO] mach={mach}, reynolds={reynolds}")


if __name__ == "__main__":
    main()