#!/usr/bin/env python3
"""
Run SU2 over geometry+flow design-space definitions.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports when invoked as `python scripts/run_design_sweep.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from su2_utils import (  # noqa: E402
    DEFAULT_MIN_RMS_DROP,
    DEFAULT_MIN_RMS_RHO_FINAL,
    RANS_MIN_RMS_DROP,
    RANS_MIN_RMS_RHO_FINAL,
    check_convergence,
    cfg_sha256,
    extract_coeff,
    get_su2_version,
    load_history_features,
    parse_cfg_cfl,
    physical_coefficients_ok,
)


ROOT = Path(__file__).resolve().parent.parent
CASE_ROOT = ROOT / "su2_cases"
DEFAULT_CFG_RANS = CASE_ROOT / "rans_NACA0012.cfg"
DEFAULT_CFG_EULER = CASE_ROOT / "inv_NACA0012.cfg"
DEFAULT_DESIGN_SPACE = ROOT / "datasets" / "raw" / "design_space.csv"
DEFAULT_RAW_DATA = ROOT / "datasets" / "raw" / "aero_design_raw.csv"

SOLVER_MESH = {
    "rans": CASE_ROOT / "mesh_NACA0012_inv.su2",
    "euler": CASE_ROOT / "mesh_NACA0012_inv.su2",
}


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


def run_case(
    case_dir: Path,
    cfg_file: Path,
    retry_max: int,
    base_cfl: float,
    require_convergence: bool,
    min_rms_rho_final: float,
    min_rms_drop: float,
    capture_last_coefficients: bool = False,
) -> dict:
    last_reason = "solver_failed"
    best_marginal: dict | None = None
    last_parsed: dict | None = None
    for attempt in range(1, retry_max + 2):
        cfg_text = cfg_file.read_text(encoding="utf-8")
        if attempt > 1:
            cfl = max(base_cfl / (2 ** (attempt - 2)), 1.0)
            cfg_text = set_cfg_value(cfg_text, "CFL_NUMBER", cfl)
            cfg_file.write_text(cfg_text, encoding="utf-8")

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
            last_reason = "nonzero_exit"
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

        if cl is None or cd is None:
            last_reason = "missing_coefficients"
            continue

        history_features = load_history_features(case_dir / "history.csv")
        if capture_last_coefficients:
            last_parsed = {
                "cl": cl,
                "cd": cd,
                "attempts": attempt,
                **history_features,
            }

        if not physical_coefficients_ok(cl, cd):
            last_reason = "physical_bounds"
            continue

        converged = check_convergence(history_features, min_rms_rho_final, min_rms_drop)
        if require_convergence and not converged:
            last_reason = "not_converged"
            candidate = {
                "status": "success",
                "attempts": attempt,
                "cl": cl,
                "cd": cd,
                "converged": False,
                "failure_reason": None,
                **history_features,
            }
            if best_marginal is None or history_features["rms_rho_final"] < best_marginal["rms_rho_final"]:
                best_marginal = candidate
            continue

        return {
            "status": "success",
            "attempts": attempt,
            "cl": cl,
            "cd": cd,
            "converged": converged,
            "failure_reason": None,
            **history_features,
        }

    if best_marginal is not None:
        return best_marginal

    if last_parsed is not None:
        return {
            "status": "failed",
            "attempts": last_parsed.get("attempts", retry_max + 1),
            "cl": last_parsed.get("cl"),
            "cd": last_parsed.get("cd"),
            "converged": False,
            "failure_reason": last_reason,
            "rms_rho_final": last_parsed.get("rms_rho_final"),
            "rms_rho_drop": last_parsed.get("rms_rho_drop"),
            "rms_nu_final": last_parsed.get("rms_nu_final"),
            "has_history": last_parsed.get("has_history"),
        }

    return {
        "status": "failed",
        "attempts": retry_max + 1,
        "cl": None,
        "cd": None,
        "converged": False,
        "failure_reason": last_reason,
        **load_history_features(case_dir / "history.csv"),
    }


def load_completed_design_ids(raw_output: Path) -> set[str]:
    if not raw_output.exists():
        return set()
    df = pd.read_csv(raw_output)
    if "design_id" not in df.columns or "status" not in df.columns:
        return set()
    ok = df[df["status"] == "success"]
    return set(ok["design_id"].astype(str))


def main() -> None:
    p = argparse.ArgumentParser(description="Run design-space SU2 sweep with metadata.")
    p.add_argument("--design-space", type=Path, default=DEFAULT_DESIGN_SPACE)
    p.add_argument("--cfg", type=Path, default=None, help="SU2 cfg template (overrides --solver).")
    p.add_argument(
        "--solver",
        choices=["rans", "euler"],
        default="rans",
        help="Solver template: rans (default, Spalart-Allmaras) or euler (legacy inviscid).",
    )
    p.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_DATA)
    p.add_argument("--retry-max", type=int, default=3)
    p.add_argument("--append-raw-output", action="store_true")
    p.add_argument("--skip-completed", action="store_true", help="Skip design_ids already successful in raw-output.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--manifest", type=Path, default=None, help="Write sweep manifest JSON (default: run dir).")
    p.add_argument(
        "--min-rms-rho-final",
        type=float,
        default=None,
        help=f"Max final log10 RMS density residual (default: {RANS_MIN_RMS_RHO_FINAL} for RANS, {DEFAULT_MIN_RMS_RHO_FINAL} for Euler).",
    )
    p.add_argument(
        "--min-rms-drop",
        type=float,
        default=None,
        help=f"Min RMS density drop (default: {RANS_MIN_RMS_DROP} for RANS, {DEFAULT_MIN_RMS_DROP} for Euler).",
    )
    p.add_argument(
        "--no-convergence-check",
        action="store_true",
        help="Accept cases without residual convergence (not recommended for RANS).",
    )
    args = p.parse_args()

    if args.min_rms_rho_final is None:
        args.min_rms_rho_final = RANS_MIN_RMS_RHO_FINAL if args.solver == "rans" else DEFAULT_MIN_RMS_RHO_FINAL
    if args.min_rms_drop is None:
        args.min_rms_drop = RANS_MIN_RMS_DROP if args.solver == "rans" else DEFAULT_MIN_RMS_DROP

    cfg_path = args.cfg or (DEFAULT_CFG_RANS if args.solver == "rans" else DEFAULT_CFG_EULER)
    mesh_file = SOLVER_MESH[args.solver]
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not mesh_file.exists():
        raise FileNotFoundError(mesh_file)

    df = pd.read_csv(args.design_space)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    completed_ids: set[str] = set()
    if args.skip_completed:
        completed_ids = load_completed_design_ids(args.raw_output)
        if completed_ids:
            print(f"[INFO] skipping {len(completed_ids)} already-successful design_ids")

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_root = CASE_ROOT / "design_runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    base_cfg_text = cfg_path.read_text(encoding="utf-8")
    base_cfl = parse_cfg_cfl(base_cfg_text)
    require_convergence = not args.no_convergence_check

    for i, row in df.iterrows():
        design_id = str(row["design_id"])
        if design_id in completed_ids:
            continue

        case_id = f"case_{i+1:04d}"
        case_dir = run_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

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
        cfg_text = set_cfg_value(cfg_text, "REYNOLDS_NUMBER", float(row["reynolds"]))
        cfg_text += (
            "\n% DESIGN_METADATA (not parsed by SU2)\n"
            f"% GEOMETRY_THICKNESS= {geometry['geometry_thickness']}\n"
            f"% GEOMETRY_CAMBER= {geometry['geometry_camber']}\n"
            f"% GEOMETRY_CAMBER_POS= {geometry['geometry_camber_pos']}\n"
        )
        cfg_case_path = case_dir / cfg_path.name
        cfg_case_path.write_text(cfg_text, encoding="utf-8")

        mesh_case_path = case_dir / mesh_file.name
        mesh_meta = deform_airfoil_mesh(
            mesh_in=mesh_file,
            mesh_out=mesh_case_path,
            thickness=float(row["geometry_thickness"]),
            camber=float(row["geometry_camber"]),
            camber_pos=float(row["geometry_camber_pos"]),
        )

        result = run_case(
            case_dir=case_dir,
            cfg_file=cfg_case_path,
            retry_max=args.retry_max,
            base_cfl=base_cfl,
            require_convergence=require_convergence,
            min_rms_rho_final=args.min_rms_rho_final,
            min_rms_drop=args.min_rms_drop,
        )

        cl, cd = result.get("cl"), result.get("cd")
        cl_cd = (cl / cd) if (cl is not None and cd not in (None, 0.0)) else None

        case_meta = {
            "design_id": design_id,
            "case_id": case_id,
            "solver": args.solver,
            "cfg_template": str(cfg_path.relative_to(ROOT)),
            "status": result["status"],
            "attempts": result["attempts"],
            "failure_reason": result.get("failure_reason"),
            "converged": result.get("converged", False),
            "cl": cl,
            "cd": cd,
            "geometry": geometry,
            "flow": {
                "aoa": float(row["aoa"]),
                "mach": float(row["mach"]),
                "reynolds": float(row["reynolds"]),
            },
            "history": {
                k: result.get(k)
                for k in (
                    "rms_rho_final",
                    "rms_rho_drop",
                    "rms_nu_final",
                    "convergence_rate",
                    "has_history",
                )
            },
        }
        (case_dir / "case_meta.json").write_text(json.dumps(case_meta, indent=2), encoding="utf-8")

        rows.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "design_id": design_id,
                "solver": args.solver,
                "geometry_thickness": row["geometry_thickness"],
                "geometry_camber": row["geometry_camber"],
                "geometry_camber_pos": row["geometry_camber_pos"],
                "aoa": row["aoa"],
                "mach": row["mach"],
                "reynolds": row["reynolds"],
                "cl": cl,
                "cd": cd,
                "cl_cd": cl_cd,
                "status": result["status"],
                "attempts": result["attempts"],
                "converged": result.get("converged", False),
                "failure_reason": result.get("failure_reason"),
                "rms_rho_final": result.get("rms_rho_final"),
                "rms_rho_drop": result.get("rms_rho_drop"),
                "rms_nu_final": result.get("rms_nu_final"),
                "has_history": result.get("has_history"),
                "airfoil_node_count": mesh_meta["airfoil_node_count"],
                "case_dir": str(case_dir.relative_to(ROOT)),
            }
        )

    if not rows:
        print("[INFO] no new cases to run (all skipped or empty design space)")
        return

    out_df = pd.DataFrame(rows)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    if args.append_raw_output and args.raw_output.exists():
        prev_df = pd.read_csv(args.raw_output)
        if list(prev_df.columns) != list(out_df.columns):
            raise ValueError(
                f"--append-raw-output column mismatch: existing has {list(prev_df.columns)}, "
                f"new run has {list(out_df.columns)}"
            )
        out_df = pd.concat([prev_df, out_df], ignore_index=True)
    out_df.to_csv(args.raw_output, index=False)

    manifest = {
        "run_id": run_id,
        "timestamp": pd.Timestamp.now().isoformat(),
        "solver": args.solver,
        "cfg_template": str(cfg_path.relative_to(ROOT)),
        "cfg_sha256": cfg_sha256(cfg_path),
        "su2_version": get_su2_version(),
        "design_space": str(args.design_space.relative_to(ROOT)),
        "n_requested": int(len(df)),
        "n_run": int(len(rows)),
        "n_success": int((out_df["status"] == "success").sum()) if "status" in out_df.columns else 0,
        "n_failed": int((out_df["status"] == "failed").sum()) if "status" in out_df.columns else 0,
        "retry_max": args.retry_max,
        "require_convergence": require_convergence,
        "convergence_thresholds": {
            "min_rms_rho_final": args.min_rms_rho_final,
            "min_rms_drop": args.min_rms_drop,
        },
        "raw_output": str(args.raw_output.relative_to(ROOT)),
        "run_root": str(run_root.relative_to(ROOT)),
    }
    manifest_path = args.manifest or (run_root / "sweep_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_success = (out_df["status"] == "success").sum()
    n_failed = (out_df["status"] == "failed").sum()
    print(f"[DONE] run_id={run_id}")
    print(f"[DONE] wrote {len(out_df)} rows to {args.raw_output}")
    print(f"[INFO] success={n_success} failed={n_failed}")
    print(f"[INFO] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
