#!/usr/bin/env python3
"""
Run SU2 over geometry+flow design-space definitions.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CASE_ROOT = ROOT / "su2_cases"
DEFAULT_CFG = CASE_ROOT / "inv_NACA0012.cfg"
DEFAULT_DESIGN_SPACE = ROOT / "datasets" / "raw" / "design_space.csv"
DEFAULT_RAW_DATA = ROOT / "datasets" / "raw" / "aero_design_raw.csv"


def set_cfg_value(cfg_text: str, key: str, value: float) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=.*$", flags=re.MULTILINE)
    new_line = f"{key}= {value}"
    if pattern.search(cfg_text):
        return pattern.sub(new_line, cfg_text)
    return cfg_text.rstrip() + f"\n{new_line}\n"


def extract_coeff(text: str, coeff: str) -> float | None:
    # Total CL: 1.234 | ...
    m = re.search(rf"^Total\s+{re.escape(coeff)}\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None


def run_case(case_dir: Path, cfg_file: Path, retry_max: int) -> tuple[str, int, float | None, float | None]:
    for attempt in range(1, retry_max + 2):
        out_file = case_dir / f"output_attempt_{attempt}.txt"
        with out_file.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                ["SU2_CFD", cfg_file.name],
                cwd=case_dir,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode != 0:
            continue

        forces_path = case_dir / "forces_breakdown.dat"
        cl = cd = None
        if forces_path.exists():
            txt = forces_path.read_text(encoding="utf-8", errors="ignore")
            cl = extract_coeff(txt, "CL")
            cd = extract_coeff(txt, "CD")
        if cl is None or cd is None:
            txt = out_file.read_text(encoding="utf-8", errors="ignore")
            cl = extract_coeff(txt, "CL")
            cd = extract_coeff(txt, "CD")

        if cl is not None and cd is not None:
            return "success", attempt, cl, cd
    return "failed", retry_max + 1, None, None


def main() -> None:
    p = argparse.ArgumentParser(description="Run design-space SU2 sweep with metadata.")
    p.add_argument("--design-space", type=Path, default=DEFAULT_DESIGN_SPACE)
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    p.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_DATA)
    p.add_argument("--retry-max", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="Optional cap for smoke tests.")
    args = p.parse_args()

    df = pd.read_csv(args.design_space)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_root = CASE_ROOT / "design_runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    base_cfg_text = args.cfg.read_text(encoding="utf-8")
    mesh_file = CASE_ROOT / "mesh_NACA0012_inv.su2"

    for i, row in df.iterrows():
        case_id = f"case_{i+1:04d}"
        case_dir = run_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Geometry metadata (current phase: param schema + provenance).
        geometry = {
            "representation": "parametric_camber_thickness",
            "geometry_thickness": float(row["geometry_thickness"]),
            "geometry_camber": float(row["geometry_camber"]),
            "geometry_camber_pos": float(row["geometry_camber_pos"]),
        }
        (case_dir / "geometry.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")

        cfg_text = base_cfg_text
        cfg_text = set_cfg_value(cfg_text, "AOA", float(row["aoa"]))
        cfg_text = set_cfg_value(cfg_text, "MACH_NUMBER", float(row["mach"]))
        # Keep Reynolds in metadata; set only if key already exists in cfg.
        if re.search(r"^\s*REYNOLDS_NUMBER\s*=", cfg_text, flags=re.MULTILINE):
            cfg_text = set_cfg_value(cfg_text, "REYNOLDS_NUMBER", float(row["reynolds"]))

        cfg_text += (
            "\n% DESIGN_METADATA (not parsed by SU2)\n"
            f"% GEOMETRY_THICKNESS= {geometry['geometry_thickness']}\n"
            f"% GEOMETRY_CAMBER= {geometry['geometry_camber']}\n"
            f"% GEOMETRY_CAMBER_POS= {geometry['geometry_camber_pos']}\n"
        )
        cfg_path = case_dir / args.cfg.name
        cfg_path.write_text(cfg_text, encoding="utf-8")

        if mesh_file.exists():
            shutil.copy2(mesh_file, case_dir / mesh_file.name)

        status, attempts, cl, cd = run_case(case_dir, cfg_path, args.retry_max)
        cl_cd = (cl / cd) if (cl is not None and cd not in (None, 0.0)) else None

        rows.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "design_id": row["design_id"],
                "geometry_thickness": row["geometry_thickness"],
                "geometry_camber": row["geometry_camber"],
                "geometry_camber_pos": row["geometry_camber_pos"],
                "aoa": row["aoa"],
                "mach": row["mach"],
                "reynolds": row["reynolds"],
                "cl": cl,
                "cd": cd,
                "cl_cd": cl_cd,
                "status": status,
                "attempts": attempts,
                "case_dir": str(case_dir.relative_to(ROOT)),
            }
        )

    out_df = pd.DataFrame(rows)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.raw_output, index=False)
    print(f"[DONE] run_id={run_id}")
    print(f"[DONE] wrote {len(out_df)} rows to {args.raw_output}")
    print(f"[INFO] success={(out_df['status'] == 'success').sum()} failed={(out_df['status'] == 'failed').sum()}")


if __name__ == "__main__":
    main()
