#!/usr/bin/env python3
"""
Convert raw SU2 case outputs into ML-ready aerodynamic dataset.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CASE_ROOT = ROOT / "su2_cases"
OUTPUT_DEFAULT = ROOT / "datasets" / "processed" / "aero_ml_dataset.csv"
CFG_DEFAULT = CASE_ROOT / "inv_NACA0012.cfg"
QUALITY_REPORT_DEFAULT = ROOT / "datasets" / "processed" / "aero_ml_quality_report.csv"


def parse_cfg_value(cfg_path: Path, key: str) -> float | None:
    if not cfg_path.exists():
        return None
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^\s%]+)")
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def extract_coeff(forces_file: Path, coeff: str) -> float | None:
    pattern = re.compile(rf"^Total\s+{re.escape(coeff)}\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")
    for line in forces_file.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            return float(match.group(1))
    return None


def load_history_features(history_path: Path) -> dict:
    default = {
        "rms_rho_final": 0.0,
        "rms_rho_u_final": 0.0,
        "rms_rho_v_final": 0.0,
        "rms_rho_e_final": 0.0,
        "rms_rho_drop": 0.0,
        "convergence_rate": 0.0,
        "has_history": 0,
    }
    if not history_path.exists():
        return default

    df = pd.read_csv(history_path)
    cols = {c.strip().strip('"'): c for c in df.columns}
    req = ["rms[Rho]", "rms[RhoU]", "rms[RhoV]", "rms[RhoE]"]
    if not all(c in cols for c in req):
        return default

    rho = pd.to_numeric(df[cols["rms[Rho]"]], errors="coerce").dropna()
    rho_u = pd.to_numeric(df[cols["rms[RhoU]"]], errors="coerce").dropna()
    rho_v = pd.to_numeric(df[cols["rms[RhoV]"]], errors="coerce").dropna()
    rho_e = pd.to_numeric(df[cols["rms[RhoE]"]], errors="coerce").dropna()
    if rho.empty:
        return default

    n = len(rho)
    x = np.arange(n, dtype=float)
    # RMS values are log10-like; fit linear slope for convergence trend.
    slope = float(np.polyfit(x, rho.to_numpy(dtype=float), 1)[0]) if n > 1 else np.nan

    return {
        "rms_rho_final": float(rho.iloc[-1]),
        "rms_rho_u_final": float(rho_u.iloc[-1]) if not rho_u.empty else 0.0,
        "rms_rho_v_final": float(rho_v.iloc[-1]) if not rho_v.empty else 0.0,
        "rms_rho_e_final": float(rho_e.iloc[-1]) if not rho_e.empty else 0.0,
        "rms_rho_drop": float(rho.iloc[0] - rho.iloc[-1]),
        "convergence_rate": float(slope) if np.isfinite(slope) else 0.0,
        "has_history": 1,
    }


def build_dataset(case_root: Path, cfg_path: Path, geometry_id: str) -> pd.DataFrame:
    mach = parse_cfg_value(cfg_path, "MACH_NUMBER")
    reynolds = parse_cfg_value(cfg_path, "REYNOLDS_NUMBER")
    mach = 0.0 if mach is None else mach
    reynolds = 0.0 if reynolds is None else reynolds

    rows: list[dict] = []
    for case_dir in sorted(case_root.glob("AoA_*")):
        if not case_dir.is_dir():
            continue
        match = re.match(r"^AoA_([+-]?\d*\.?\d+)$", case_dir.name)
        if not match:
            continue
        aoa = float(match.group(1))

        forces_file = case_dir / "forces_breakdown.dat"
        if not forces_file.exists():
            continue
        cl = extract_coeff(forces_file, "CL")
        cd = extract_coeff(forces_file, "CD")
        if cl is None or cd is None:
            continue

        hist_features = load_history_features(case_dir / "history.csv")
        rows.append(
            {
                "geometry_id": geometry_id,
                "aoa": aoa,
                "aoa_squared": aoa**2,
                "mach": mach,
                "reynolds": reynolds,
                "cl": cl,
                "cd": cd,
                "cl_cd": (cl / cd) if cd != 0 else np.nan,
                **hist_features,
                "case_dir": str(case_dir.relative_to(ROOT)),
            }
        )

    if not rows:
        raise RuntimeError("No valid AoA_* cases with parseable CL/CD found under su2_cases.")
    df = pd.DataFrame(rows).sort_values("aoa").reset_index(drop=True)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    numeric_cols = [
        "aoa",
        "aoa_squared",
        "cl",
        "cd",
        "cl_cd",
        "rms_rho_final",
        "rms_rho_u_final",
        "rms_rho_v_final",
        "rms_rho_e_final",
        "rms_rho_drop",
        "convergence_rate",
        "has_history",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isna().any().any():
        bad = df[df[numeric_cols].isna().any(axis=1)]
        raise ValueError(f"NaNs found in processed dataset rows:\n{bad[['aoa','cl','cd']].to_string(index=False)}")

    if not np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values detected in numeric dataset columns.")

    if len(df) != df["aoa"].nunique():
        raise ValueError("Inconsistent row counts: duplicate AoA values found.")


def quality_gate(
    df: pd.DataFrame,
    min_rms_rho_final: float,
    min_rms_drop: float,
    max_abs_cl: float,
    min_cd: float,
) -> pd.DataFrame:
    checks = pd.DataFrame(index=df.index)
    checks["pass_cd_positive"] = df["cd"] >= min_cd
    checks["pass_cl_physical"] = df["cl"].abs() <= max_abs_cl
    checks["pass_history"] = df["has_history"] == 1
    checks["pass_converged_residual"] = df["rms_rho_final"] <= min_rms_rho_final
    checks["pass_residual_drop"] = df["rms_rho_drop"] >= min_rms_drop
    checks["quality_pass"] = checks.all(axis=1)

    merged = pd.concat([df.copy(), checks], axis=1)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed ML dataset from SU2 outputs.")
    parser.add_argument("--case-root", type=Path, default=CASE_ROOT)
    parser.add_argument("--cfg", type=Path, default=CFG_DEFAULT)
    parser.add_argument("--geometry-id", default="NACA0012")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--quality-report", type=Path, default=QUALITY_REPORT_DEFAULT)
    parser.add_argument("--min-rms-rho-final", type=float, default=-6.0)
    parser.add_argument("--min-rms-drop", type=float, default=4.0)
    parser.add_argument("--max-abs-cl", type=float, default=3.0)
    parser.add_argument("--min-cd", type=float, default=1e-6)
    parser.add_argument("--allow-failed-quality", action="store_true")
    args = parser.parse_args()

    df = build_dataset(args.case_root, args.cfg, args.geometry_id)
    validate_dataset(df)
    quality_df = quality_gate(
        df=df,
        min_rms_rho_final=args.min_rms_rho_final,
        min_rms_drop=args.min_rms_drop,
        max_abs_cl=args.max_abs_cl,
        min_cd=args.min_cd,
    )

    passed = quality_df[quality_df["quality_pass"]].copy()
    failed = quality_df[~quality_df["quality_pass"]].copy()
    if not args.allow_failed_quality and len(failed) > 0:
        display_cols = ["aoa", "cl", "cd", "rms_rho_final", "rms_rho_drop", "quality_pass"]
        raise ValueError(
            "Quality gate failed for one or more cases. "
            "Review datasets/processed/aero_ml_quality_report.csv\n"
            f"{failed[display_cols].to_string(index=False)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed.to_csv(args.output, index=False)
    quality_df.to_csv(args.quality_report, index=False)
    print(f"[DONE] wrote {len(passed)} quality-passed rows to {args.output}")
    print(f"[DONE] quality report saved to {args.quality_report}")
    print(f"[INFO] quality passed={len(passed)} failed={len(failed)}")


if __name__ == "__main__":
    main()
