#!/usr/bin/env python3
"""
Convert raw SU2 case outputs into ML-ready aerodynamic dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from su2_utils import (  # noqa: E402
    DEFAULT_MIN_RMS_DROP,
    DEFAULT_MIN_RMS_RHO_FINAL,
    MAX_ABS_CL,
    MIN_CD,
    RANS_MIN_RMS_DROP,
    RANS_MIN_RMS_RHO_FINAL,
    load_history_features,
)


ROOT = Path(__file__).resolve().parent.parent
CASE_ROOT = ROOT / "su2_cases"
OUTPUT_DEFAULT = ROOT / "datasets" / "processed" / "aero_ml_dataset.csv"
CFG_DEFAULT = CASE_ROOT / "rans_NACA0012.cfg"
QUALITY_REPORT_DEFAULT = ROOT / "datasets" / "processed" / "aero_ml_quality_report.csv"
DESIGN_RAW_DEFAULT = ROOT / "datasets" / "raw" / "aero_design_raw.csv"
DESIGN_OUTPUT_DEFAULT = ROOT / "datasets" / "processed" / "aero_design_dataset.csv"
DESIGN_QUALITY_REPORT_DEFAULT = ROOT / "datasets" / "processed" / "aero_design_quality_report.csv"


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


def load_case_meta(case_dir: Path) -> dict:
    meta_path = case_dir / "case_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def extract_coeff(forces_file: Path, coeff: str) -> float | None:
    pattern = re.compile(rf"^Total\s+{re.escape(coeff)}\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")
    for line in forces_file.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            return float(match.group(1))
    return None


def build_dataset(case_root: Path, cfg_path: Path, geometry_id: str) -> pd.DataFrame:
    mach = parse_cfg_value(cfg_path, "MACH_NUMBER")
    reynolds = parse_cfg_value(cfg_path, "REYNOLDS_NUMBER")
    mach = 0.0 if mach is None else mach
    reynolds = 0.0 if reynolds is None else reynolds

    rows: list[dict] = []
    run_case_dirs = {
        path.parent
        for path in case_root.glob("runs/*/cases/*/forces_breakdown.dat")
    }
    legacy_case_dirs = {path for path in case_root.glob("AoA_*") if path.is_dir()}
    case_dirs = sorted(run_case_dirs.union(legacy_case_dirs))

    for case_dir in case_dirs:
        case_meta = load_case_meta(case_dir)
        if case_meta:
            if case_meta.get("status") != "success":
                continue
            aoa = float(case_meta.get("aoa", 0.0))
            case_mach = float(case_meta.get("mach", mach))
            case_re = float(case_meta.get("reynolds", reynolds))
            run_id = case_meta.get("run_id", "legacy")
            case_id = case_meta.get("case_id", case_dir.name)
        else:
            match = re.match(r"^AoA_([+-]?\d*\.?\d+)$", case_dir.name)
            if not match:
                continue
            aoa = float(match.group(1))
            case_mach = mach
            case_re = reynolds
            run_id = "legacy"
            case_id = case_dir.name

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
                "run_id": run_id,
                "case_id": case_id,
                "aoa": aoa,
                "aoa_squared": aoa**2,
                "mach": case_mach,
                "reynolds": case_re,
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

    if "run_id" in df.columns and "case_id" in df.columns:
        if len(df) != len(df[["run_id", "case_id"]].drop_duplicates()):
            raise ValueError("Inconsistent row counts: duplicate case identifiers found.")


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


def design_quality_gate(
    df: pd.DataFrame,
    min_rms_rho_final: float,
    min_rms_drop: float,
    max_abs_cl: float,
    min_cd: float,
) -> pd.DataFrame:
    checks = pd.DataFrame(index=df.index)
    checks["pass_status_success"] = df["status"] == "success"
    checks["pass_cd_positive"] = df["cd"] >= min_cd
    checks["pass_cl_physical"] = df["cl"].abs() <= max_abs_cl
    if "has_history" in df.columns and (df["has_history"].fillna(0).astype(int) == 1).any():
        checks["pass_history"] = df["has_history"].fillna(0).astype(int) == 1
    else:
        checks["pass_history"] = True
    if "rms_rho_final" in df.columns and df["rms_rho_final"].notna().any():
        checks["pass_converged_residual"] = df["rms_rho_final"] <= min_rms_rho_final
        checks["pass_residual_drop"] = df["rms_rho_drop"] >= min_rms_drop
    else:
        checks["pass_converged_residual"] = True
        checks["pass_residual_drop"] = True
    if "converged" in df.columns and df["converged"].notna().any():
        checks["pass_converged_flag"] = df["converged"].fillna(False).astype(bool)
    else:
        checks["pass_converged_flag"] = True

    # Base gate: successful SU2 run with physically plausible coefficients.
    checks["quality_pass"] = (
        checks["pass_status_success"]
        & checks["pass_cd_positive"]
        & checks["pass_cl_physical"]
        & checks["pass_history"]
    )
    # Strict tier for high-confidence training subsets (convergence + residuals).
    checks["strict_pass"] = checks.all(axis=1)
    return pd.concat([df.copy(), checks], axis=1)


def build_design_dataset(raw_csv_paths: list[Path]) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for raw_csv in raw_csv_paths:
        if not raw_csv.exists():
            raise FileNotFoundError(raw_csv)
        dfs.append(pd.read_csv(raw_csv))
    df = pd.concat(dfs, ignore_index=True)

    dup_subset = ["geometry_thickness", "geometry_camber", "geometry_camber_pos", "aoa", "mach", "reynolds"]
    missing_dup = [c for c in dup_subset if c not in df.columns]
    if not missing_dup:
        before = len(df)
        df = df.drop_duplicates(subset=dup_subset, keep="last")
        dropped = before - len(df)
        if dropped:
            print(f"[INFO] dropped {dropped} duplicate design/flow rows (kept newest occurrence)")
    required = [
        "geometry_thickness",
        "geometry_camber",
        "geometry_camber_pos",
        "aoa",
        "mach",
        "reynolds",
        "cl",
        "cd",
        "status",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Design raw dataset missing columns: {missing}")

    for col in ["geometry_thickness", "geometry_camber", "geometry_camber_pos", "aoa", "mach", "reynolds", "cl", "cd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "converged" not in df.columns:
        df["converged"] = np.nan
    if "has_history" not in df.columns:
        df["has_history"] = 0
    if "rms_rho_final" not in df.columns:
        df["rms_rho_final"] = np.nan
    if "rms_rho_drop" not in df.columns:
        df["rms_rho_drop"] = np.nan
    df["cl_cd"] = df["cl"] / df["cd"]
    df["geometry_param_1"] = df["geometry_thickness"]
    df["geometry_param_2"] = df["geometry_camber"]
    df["geometry_param_3"] = df["geometry_camber_pos"]
    df["AoA"] = df["aoa"]
    df["Mach"] = df["mach"]
    df["Re"] = df["reynolds"]
    df["CL"] = df["cl"]
    df["CD"] = df["cd"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed ML dataset from SU2 outputs.")
    parser.add_argument("--dataset-type", choices=["aero", "design"], default="aero")
    parser.add_argument("--case-root", type=Path, default=CASE_ROOT)
    parser.add_argument("--cfg", type=Path, default=CFG_DEFAULT)
    parser.add_argument("--geometry-id", default="NACA0012")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--quality-report", type=Path, default=QUALITY_REPORT_DEFAULT)
    parser.add_argument(
        "--design-raw",
        type=Path,
        nargs="+",
        default=[DESIGN_RAW_DEFAULT],
        help="One or more raw design sweep CSVs to merge before processing.",
    )
    parser.add_argument("--design-output", type=Path, default=DESIGN_OUTPUT_DEFAULT)
    parser.add_argument("--design-quality-report", type=Path, default=DESIGN_QUALITY_REPORT_DEFAULT)
    parser.add_argument(
        "--min-rms-rho-final",
        type=float,
        default=None,
        help="Strict-tier max final log10 RMS density residual.",
    )
    parser.add_argument(
        "--min-rms-drop",
        type=float,
        default=None,
        help="Strict-tier min RMS density drop.",
    )
    parser.add_argument("--max-abs-cl", type=float, default=MAX_ABS_CL)
    parser.add_argument("--min-cd", type=float, default=MIN_CD)
    parser.add_argument(
        "--require-strict-convergence",
        action="store_true",
        help="Only keep rows passing full convergence/residual checks (strict_pass).",
    )
    parser.add_argument(
        "--allow-failed-quality",
        action="store_true",
        help="Include quality-failed rows in the output dataset (not recommended).",
    )
    parser.add_argument(
        "--fail-on-rejects",
        action="store_true",
        help="Abort if any row fails the active quality gate (default: exclude rejects and continue).",
    )
    args = parser.parse_args()

    if args.min_rms_rho_final is None:
        args.min_rms_rho_final = (
            RANS_MIN_RMS_RHO_FINAL if args.dataset_type == "design" else DEFAULT_MIN_RMS_RHO_FINAL
        )
    if args.min_rms_drop is None:
        args.min_rms_drop = RANS_MIN_RMS_DROP if args.dataset_type == "design" else DEFAULT_MIN_RMS_DROP

    if args.dataset_type == "design":
        raw_df = build_design_dataset(list(args.design_raw))
        quality_df = design_quality_gate(
            df=raw_df,
            min_rms_rho_final=args.min_rms_rho_final,
            min_rms_drop=args.min_rms_drop,
            max_abs_cl=args.max_abs_cl,
            min_cd=args.min_cd,
        )
        gate_col = "strict_pass" if args.require_strict_convergence else "quality_pass"
        passed = quality_df[quality_df[gate_col]].copy()
        failed = quality_df[~quality_df[gate_col]].copy()
        strict_n = int(quality_df["strict_pass"].sum()) if "strict_pass" in quality_df.columns else 0

        if passed.empty:
            raise RuntimeError(
                f"No design rows passed {'strict' if args.require_strict_convergence else 'base'} quality gate. "
                f"Review {args.design_quality_report} (failed={len(failed)})."
            )
        if args.allow_failed_quality:
            passed = quality_df.copy()

        if args.fail_on_rejects and len(failed) > 0:
            display_cols = [
                c
                for c in [
                    "design_id",
                    "status",
                    "cl",
                    "cd",
                    "converged",
                    "rms_rho_final",
                    "quality_pass",
                    "strict_pass",
                ]
                if c in failed.columns
            ]
            raise ValueError(
                "Design quality gate failed for one or more cases. "
                f"Review {args.design_quality_report}\n"
                f"{failed[display_cols].head(10).to_string(index=False)}"
            )

        if len(failed) > 0:
            print(
                f"[WARN] excluded {len(failed)} rejected rows from dataset "
                f"(see {args.design_quality_report})"
            )

        design_df = passed[
            [
                "geometry_param_1",
                "geometry_param_2",
                "geometry_param_3",
                "AoA",
                "Mach",
                "Re",
                "CL",
                "CD",
                "cl_cd",
            ]
        ].copy()
        numeric_cols = list(design_df.columns)
        if design_df[numeric_cols].isna().any().any():
            raise ValueError("NaN values found in design dataset numeric fields.")
        if not np.isfinite(design_df[numeric_cols].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite values found in design dataset numeric fields.")

        args.design_output.parent.mkdir(parents=True, exist_ok=True)
        design_df.to_csv(args.design_output, index=False)
        quality_df.to_csv(args.design_quality_report, index=False)
        print(f"[DONE] wrote {len(design_df)} rows to {args.design_output} (gate={gate_col})")
        print(f"[DONE] design quality report -> {args.design_quality_report}")
        print(
            f"[INFO] gate passed={len(passed)} failed={len(failed)} "
            f"(base quality_pass={int(quality_df['quality_pass'].sum())}, strict_pass={strict_n})"
        )
        return

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
