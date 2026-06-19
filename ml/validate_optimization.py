#!/usr/bin/env python3
"""
Re-run SU2 on optimized designs and compare surrogate predictions to fresh CFD.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_design_sweep import (  # noqa: E402
    SOLVER_MESH,
    deform_airfoil_mesh,
    run_case,
    set_cfg_value,
)
from su2_utils import (  # noqa: E402
    RANS_MIN_RMS_DROP,
    RANS_MIN_RMS_RHO_FINAL,
    parse_cfg_cfl,
)

DEFAULT_OPT_RESULT = ROOT / "results" / "design_optimization_result.json"
DEFAULT_MODEL = ROOT / "results" / "models" / "design_rf_model.joblib"
DEFAULT_CFG_RANS = ROOT / "su2_cases" / "rans_NACA0012.cfg"
DEFAULT_OUT_JSON = ROOT / "results" / "optimization_validation.json"
DEFAULT_OUT_CSV = ROOT / "results" / "optimization_validation.csv"


def predict_surrogate(bundle: dict, geometry_flow: dict) -> dict[str, float]:
    from design_feature_utils import augment_design_inputs_v1

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    target_cols = bundle["target_cols"]
    augment_version = int(bundle.get("augment_version", 0))

    base_x = np.array(
        [
            [
                geometry_flow["geometry_thickness"],
                geometry_flow["geometry_camber"],
                geometry_flow["geometry_camber_pos"],
                geometry_flow["aoa"],
                geometry_flow["mach"],
                geometry_flow["reynolds"],
            ]
        ],
        dtype=float,
    )
    if augment_version > 0:
        x, _ = augment_design_inputs_v1(base_x)
    else:
        x = base_x

    pred = model.predict(x)[0]
    out = {target_cols[i]: float(pred[i]) for i in range(len(target_cols))}
    if "CL" in out and "CD" in out and out["CD"] != 0:
        out["CL_CD"] = out["CL"] / out["CD"]
    return out


def run_cfd_validation(
    geometry_flow: dict,
    cfg_path: Path,
    solver: str,
    retry_max: int,
    require_convergence: bool,
) -> dict:
    case_dir = ROOT / "su2_cases" / "validation_runs" / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    case_dir.mkdir(parents=True, exist_ok=True)

    mesh_file = SOLVER_MESH[solver]
    base_cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_text = base_cfg_text
    cfg_text = set_cfg_value(cfg_text, "AOA", float(geometry_flow["aoa"]))
    cfg_text = set_cfg_value(cfg_text, "MACH_NUMBER", float(geometry_flow["mach"]))
    cfg_text = set_cfg_value(cfg_text, "REYNOLDS_NUMBER", float(geometry_flow["reynolds"]))
    cfg_case_path = case_dir / cfg_path.name
    cfg_case_path.write_text(cfg_text, encoding="utf-8")

    mesh_case_path = case_dir / mesh_file.name
    deform_airfoil_mesh(
        mesh_in=mesh_file,
        mesh_out=mesh_case_path,
        thickness=float(geometry_flow["geometry_thickness"]),
        camber=float(geometry_flow["geometry_camber"]),
        camber_pos=float(geometry_flow["geometry_camber_pos"]),
    )

    result = run_case(
        case_dir=case_dir,
        cfg_file=cfg_case_path,
        retry_max=retry_max,
        base_cfl=parse_cfg_cfl(base_cfg_text),
        require_convergence=require_convergence,
        min_rms_rho_final=RANS_MIN_RMS_RHO_FINAL,
        min_rms_drop=RANS_MIN_RMS_DROP,
        capture_last_coefficients=True,
    )
    result["case_dir"] = str(case_dir.relative_to(ROOT))
    return result


def _within_training_hull(geometry_flow: dict) -> dict | None:
    data_path = ROOT / "datasets" / "processed" / "aero_design_dataset.csv"
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path)
    col_map = {
        "geometry_thickness": "geometry_param_1",
        "geometry_camber": "geometry_param_2",
        "geometry_camber_pos": "geometry_param_3",
        "aoa": "AoA",
        "mach": "Mach",
        "reynolds": "Re",
    }
    per_dim: dict[str, bool] = {}
    for key, col in col_map.items():
        if col not in df.columns:
            continue
        val = float(geometry_flow[key])
        per_dim[key] = bool(df[col].min() <= val <= df[col].max())
    if not per_dim:
        return None
    return {"per_dimension": per_dim, "all_inside": all(per_dim.values())}


def pct_error(predicted: float | None, actual: float | None) -> float | None:
    if predicted is None or actual is None:
        return None
    denom = max(abs(actual), 1e-12)
    return float(100.0 * abs(predicted - actual) / denom)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate optimization results with fresh SU2 runs.")
    p.add_argument("--opt-result", type=Path, default=DEFAULT_OPT_RESULT)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG_RANS)
    p.add_argument("--solver", choices=["rans", "euler"], default="rans")
    p.add_argument("--retry-max", type=int, default=3)
    p.add_argument(
        "--require-convergence",
        action="store_true",
        help="Reject validation unless residual convergence passes (default: accept marginal SU2 runs).",
    )
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    args = p.parse_args()

    opt = json.loads(args.opt_result.read_text(encoding="utf-8"))
    bundle = joblib.load(args.model)
    geometry_flow = opt["best_geometry_flow"]
    predicted = predict_surrogate(bundle, geometry_flow)

    cfd = run_cfd_validation(
        geometry_flow=geometry_flow,
        cfg_path=args.cfg,
        solver=args.solver,
        retry_max=args.retry_max,
        require_convergence=args.require_convergence,
    )

    report = {
        "objective": opt.get("objective"),
        "geometry_flow": geometry_flow,
        "predicted": predicted,
        "cfd": {
            "CL": cfd.get("cl"),
            "CD": cfd.get("cd"),
            "CL_CD": (cfd.get("cl") / cfd.get("cd")) if cfd.get("cl") is not None and cfd.get("cd") not in (None, 0) else None,
            "status": cfd.get("status"),
            "converged": cfd.get("converged"),
            "attempts": cfd.get("attempts"),
            "failure_reason": cfd.get("failure_reason"),
            "case_dir": cfd.get("case_dir"),
        },
        "within_training_hull": _within_training_hull(geometry_flow),
        "errors": {
            "CL_pct_error": pct_error(predicted.get("CL"), cfd.get("cl")),
            "CD_pct_error": pct_error(predicted.get("CD"), cfd.get("cd")),
        },
        "solver": args.solver,
        "cfg_template": str(args.cfg.relative_to(ROOT)),
    }

    row = {
        "predicted_CL": predicted.get("CL"),
        "predicted_CD": predicted.get("CD"),
        "cfd_CL": cfd.get("cl"),
        "cfd_CD": cfd.get("cd"),
        "CL_pct_error": report["errors"]["CL_pct_error"],
        "CD_pct_error": report["errors"]["CD_pct_error"],
        "cfd_status": cfd.get("status"),
        "cfd_converged": cfd.get("converged"),
        "solver": args.solver,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([row]).to_csv(args.out_csv, index=False)

    print(f"[DONE] validation JSON -> {args.out_json}")
    print(f"[DONE] validation CSV  -> {args.out_csv}")
    print(
        f"[INFO] predicted CL={predicted.get('CL'):.6f} CD={predicted.get('CD'):.6f} | "
        f"CFD CL={cfd.get('cl')} CD={cfd.get('cd')} | "
        f"errors CL={report['errors']['CL_pct_error']}% CD={report['errors']['CD_pct_error']}%"
    )
    hull = report.get("within_training_hull") or {}
    inside = hull.get("all_inside")
    print(f"[SUMMARY] within_training_hull={inside}")


if __name__ == "__main__":
    main()
