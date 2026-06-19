#!/usr/bin/env python3
"""Run Euler vs. RANS ablation on a fixed subset of design points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_design_sweep import (  # noqa: E402
    DEFAULT_CFG_EULER,
    DEFAULT_CFG_RANS,
    SOLVER_MESH,
    deform_airfoil_mesh,
    run_case,
    set_cfg_value,
)
from su2_utils import (  # noqa: E402
    DEFAULT_MIN_RMS_DROP,
    DEFAULT_MIN_RMS_RHO_FINAL,
    RANS_MIN_RMS_DROP,
    RANS_MIN_RMS_RHO_FINAL,
    parse_cfg_cfl,
)

DEFAULT_DESIGN_SPACE = ROOT / "datasets" / "raw" / "design_space.csv"
DEFAULT_OUT = ROOT / "results" / "euler_vs_rans_comparison.csv"


def run_point(row: pd.Series, solver: str, case_root: Path, retry_max: int) -> dict:
    cfg_path = DEFAULT_CFG_RANS if solver == "rans" else DEFAULT_CFG_EULER
    mesh_file = SOLVER_MESH[solver]
    case_dir = case_root / f"{row['design_id']}_{solver}"
    case_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_text = set_cfg_value(base_cfg_text, "AOA", float(row["aoa"]))
    cfg_text = set_cfg_value(cfg_text, "MACH_NUMBER", float(row["mach"]))
    cfg_text = set_cfg_value(cfg_text, "REYNOLDS_NUMBER", float(row["reynolds"]))
    cfg_case_path = case_dir / cfg_path.name
    cfg_case_path.write_text(cfg_text, encoding="utf-8")

    mesh_case_path = case_dir / mesh_file.name
    deform_airfoil_mesh(
        mesh_in=mesh_file,
        mesh_out=mesh_case_path,
        thickness=float(row["geometry_thickness"]),
        camber=float(row["geometry_camber"]),
        camber_pos=float(row["geometry_camber_pos"]),
    )

    if solver == "rans":
        min_rms_rho_final = RANS_MIN_RMS_RHO_FINAL
        min_rms_drop = RANS_MIN_RMS_DROP
    else:
        min_rms_rho_final = DEFAULT_MIN_RMS_RHO_FINAL
        min_rms_drop = DEFAULT_MIN_RMS_DROP

    result = run_case(
        case_dir=case_dir,
        cfg_file=cfg_case_path,
        retry_max=retry_max,
        base_cfl=parse_cfg_cfl(base_cfg_text),
        require_convergence=True,
        min_rms_rho_final=min_rms_rho_final,
        min_rms_drop=min_rms_drop,
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Euler vs. RANS ablation on design-space samples.")
    p.add_argument("--design-space", type=Path, default=DEFAULT_DESIGN_SPACE)
    p.add_argument("--n-points", type=int, default=20)
    p.add_argument("--retry-max", type=int, default=3)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.design_space).head(args.n_points).copy()
    run_root = ROOT / "su2_cases" / "euler_ablation" / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for _, row in df.iterrows():
        for solver in ("euler", "rans"):
            print(f"[INFO] {row['design_id']} solver={solver}")
            result = run_point(row, solver, run_root, args.retry_max)
            rows.append(
                {
                    "design_id": row["design_id"],
                    "solver": solver,
                    "geometry_thickness": row["geometry_thickness"],
                    "geometry_camber": row["geometry_camber"],
                    "geometry_camber_pos": row["geometry_camber_pos"],
                    "aoa": row["aoa"],
                    "mach": row["mach"],
                    "reynolds": row["reynolds"],
                    "CL": result.get("cl"),
                    "CD": result.get("cd"),
                    "status": result.get("status"),
                    "converged": result.get("converged"),
                    "case_dir": str((run_root / f"{row['design_id']}_{solver}").relative_to(ROOT)),
                }
            )

    out_df = pd.DataFrame(rows)
    wide = out_df.pivot_table(
        index=["design_id", "aoa", "mach", "reynolds"],
        columns="solver",
        values=["CL", "CD"],
        aggfunc="first",
    )
    wide.columns = [f"{coef}_{solver}" for coef, solver in wide.columns]
    wide = wide.reset_index()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.out, index=False)
    (args.out.with_suffix(".manifest.json")).write_text(
        json.dumps({"n_points": args.n_points, "run_root": str(run_root.relative_to(ROOT))}, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] comparison -> {args.out}")
    print(f"[SUMMARY] N={len(df)} design points | {len(rows)} solver runs")


if __name__ == "__main__":
    main()
