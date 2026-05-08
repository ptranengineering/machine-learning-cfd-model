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
import numpy as np


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


def parse_su2_mesh(mesh_path: Path) -> tuple[list[str], int, int, set[int]]:
    lines = mesh_path.read_text(encoding="utf-8").splitlines()

    npoin_idx = None
    npoin = None
    for i, line in enumerate(lines):
        if line.startswith("NPOIN="):
            npoin_idx = i
            npoin = int(line.split("=")[1].strip())
            break
    if npoin_idx is None or npoin is None:
        raise ValueError(f"Could not parse NPOIN in {mesh_path}")

    point_start = npoin_idx + 1
    point_end = point_start + npoin

    airfoil_nodes: set[int] = set()
    i = point_end
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MARKER_TAG="):
            tag = line.split("=")[1].strip()
            elems_line = lines[i + 1].strip()
            if not elems_line.startswith("MARKER_ELEMS="):
                raise ValueError("Malformed marker block")
            n_elem = int(elems_line.split("=")[1].strip())
            start = i + 2
            end = start + n_elem
            if tag == "airfoil":
                for j in range(start, end):
                    parts = lines[j].split()
                    # Edge element format: 3 node_i node_j
                    if len(parts) >= 3:
                        airfoil_nodes.add(int(parts[1]))
                        airfoil_nodes.add(int(parts[2]))
            i = end
        else:
            i += 1

    if not airfoil_nodes:
        raise ValueError(f"No airfoil marker nodes found in {mesh_path}")
    return lines, point_start, point_end, airfoil_nodes


def naca_camber_line(x: np.ndarray, m: float, p: float) -> np.ndarray:
    p = float(np.clip(p, 1e-3, 0.999))
    yc = np.where(
        x < p,
        m / (p**2) * (2 * p * x - x**2),
        m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x**2),
    )
    return yc


def naca_thickness_dist(x: np.ndarray, t: float) -> np.ndarray:
    # Classic NACA thickness polynomial.
    return 5 * t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-8))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )


def deform_airfoil_mesh(
    mesh_in: Path,
    mesh_out: Path,
    thickness: float,
    camber: float,
    camber_pos: float,
) -> dict:
    lines, p_start, p_end, airfoil_nodes = parse_su2_mesh(mesh_in)

    coords = {}
    for li in range(p_start, p_end):
        parts = lines[li].split()
        x, y, idx = float(parts[0]), float(parts[1]), int(parts[2])
        coords[idx] = (x, y, li)

    airfoil_xy = np.array([[coords[n][0], coords[n][1]] for n in sorted(airfoil_nodes)], dtype=float)
    x_min = float(np.min(airfoil_xy[:, 0]))
    x_max = float(np.max(airfoil_xy[:, 0]))
    chord = max(x_max - x_min, 1e-8)

    # Use original sign around y=0 as upper/lower discriminator.
    for n in airfoil_nodes:
        x, y, li = coords[n]
        x_norm = np.clip((x - x_min) / chord, 0.0, 1.0)
        yc = float(naca_camber_line(np.array([x_norm]), camber, camber_pos)[0]) * chord
        yt = float(naca_thickness_dist(np.array([x_norm]), thickness)[0]) * chord
        sign = 1.0 if y >= 0.0 else -1.0
        y_new = yc + sign * yt
        lines[li] = f"\t{x:.15e}\t{y_new:.15e}\t{n}"

    mesh_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "airfoil_node_count": int(len(airfoil_nodes)),
        "x_min": x_min,
        "x_max": x_max,
        "chord": chord,
    }


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
            # Coarse physical sanity checks for robust dataset quality.
            if cd <= 0 or cd > 2.0 or abs(cl) > 3.0:
                continue
            return "success", attempt, cl, cd
    return "failed", retry_max + 1, None, None


def main() -> None:
    p = argparse.ArgumentParser(description="Run design-space SU2 sweep with metadata.")
    p.add_argument("--design-space", type=Path, default=DEFAULT_DESIGN_SPACE)
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    p.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_DATA)
    p.add_argument(
        "--retry-max",
        type=int,
        default=3,
        help="Retries per case (beyond first attempt) before marking failed.",
    )
    p.add_argument(
        "--append-raw-output",
        action="store_true",
        help="Append rows to raw-output CSV instead of overwriting (same columns expected).",
    )
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

        if not mesh_file.exists():
            raise FileNotFoundError(mesh_file)
        mesh_case_path = case_dir / mesh_file.name
        mesh_meta = deform_airfoil_mesh(
            mesh_in=mesh_file,
            mesh_out=mesh_case_path,
            thickness=float(row["geometry_thickness"]),
            camber=float(row["geometry_camber"]),
            camber_pos=float(row["geometry_camber_pos"]),
        )

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
                "airfoil_node_count": mesh_meta["airfoil_node_count"],
                "case_dir": str(case_dir.relative_to(ROOT)),
            }
        )

    out_df = pd.DataFrame(rows)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    if args.append_raw_output and args.raw_output.exists():
        prev_df = pd.read_csv(args.raw_output)
        cols_prev = list(prev_df.columns)
        cols_new = list(out_df.columns)
        if cols_prev != cols_new:
            raise ValueError(
                f"--append-raw-output column mismatch: existing has {cols_prev}, new run has {cols_new}"
            )
        out_df = pd.concat([prev_df, out_df], ignore_index=True)
    out_df.to_csv(args.raw_output, index=False)
    print(f"[DONE] run_id={run_id}")
    print(f"[DONE] wrote {len(out_df)} rows to {args.raw_output}")
    print(f"[INFO] success={(out_df['status'] == 'success').sum()} failed={(out_df['status'] == 'failed').sum()}")


if __name__ == "__main__":
    main()
