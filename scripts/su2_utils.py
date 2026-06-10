"""
Shared SU2 helpers for design-space sweep and dataset building.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# Physical coefficient bounds (shared between sweep and dataset builder).
MIN_CD = 0.002
MAX_CD = 2.0
MAX_ABS_CL = 3.0

# Default convergence thresholds (log10 RMS density residual).
DEFAULT_MIN_RMS_RHO_FINAL = -6.0
DEFAULT_MIN_RMS_DROP = 4.0
# Softer defaults for transonic RANS on coarse design-sweep meshes.
RANS_MIN_RMS_RHO_FINAL = -4.0
RANS_MIN_RMS_DROP = 2.0


def get_su2_version() -> str:
    """Best-effort SU2 version string for sweep manifests."""
    try:
        proc = subprocess.run(
            ["SU2_CFD", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        text = (proc.stdout + proc.stderr).strip()
        for line in text.splitlines():
            if "SU2" in line:
                return line.strip()[:120]
        return text.splitlines()[0][:120] if text else "SU2_CFD"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "unknown"


def cfg_sha256(cfg_path: Path) -> str:
    return hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16]


def parse_cfg_cfl(cfg_text: str) -> float:
    m = re.search(r"^\s*CFL_NUMBER\s*=\s*([^\s%]+)", cfg_text, flags=re.MULTILINE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 50.0


def load_history_features(history_path: Path) -> dict:
    default = {
        "rms_rho_final": 0.0,
        "rms_rho_u_final": 0.0,
        "rms_rho_v_final": 0.0,
        "rms_rho_e_final": 0.0,
        "rms_nu_final": 0.0,
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

    nu_final = 0.0
    for nu_key in ("rms[NuTilde]", "rms[Nu_Tilde]", "RMS_NuTilde", "rms[k]", "rms[omega]"):
        if nu_key in cols:
            nu = pd.to_numeric(df[cols[nu_key]], errors="coerce").dropna()
            if not nu.empty:
                nu_final = float(nu.iloc[-1])
                break

    n = len(rho)
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, rho.to_numpy(dtype=float), 1)[0]) if n > 1 else np.nan

    return {
        "rms_rho_final": float(rho.iloc[-1]),
        "rms_rho_u_final": float(rho_u.iloc[-1]) if not rho_u.empty else 0.0,
        "rms_rho_v_final": float(rho_v.iloc[-1]) if not rho_v.empty else 0.0,
        "rms_rho_e_final": float(rho_e.iloc[-1]) if not rho_e.empty else 0.0,
        "rms_nu_final": nu_final,
        "rms_rho_drop": float(rho.iloc[0] - rho.iloc[-1]),
        "convergence_rate": float(slope) if np.isfinite(slope) else 0.0,
        "has_history": 1,
    }


def check_convergence(
    history_features: dict,
    min_rms_rho_final: float = DEFAULT_MIN_RMS_RHO_FINAL,
    min_rms_drop: float = DEFAULT_MIN_RMS_DROP,
) -> bool:
    if history_features.get("has_history") != 1:
        return False
    if history_features["rms_rho_final"] > min_rms_rho_final:
        return False
    if history_features["rms_rho_drop"] < min_rms_drop:
        return False
    return True


def physical_coefficients_ok(cl: float, cd: float) -> bool:
    return cd > MIN_CD and cd <= MAX_CD and abs(cl) <= MAX_ABS_CL


def extract_coeff(text: str, coeff: str) -> float | None:
    m = re.search(
        rf"^Total\s+{re.escape(coeff)}\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        text,
        flags=re.MULTILINE,
    )
    return float(m.group(1)) if m else None


def persist_case_artifacts(case_dir: Path) -> None:
    """Copy SU2 outputs into case_dir with stable names (history, forces)."""
    for name in ("history.csv", "forces_breakdown.dat", "surface_flow.csv"):
        src = case_dir / name
        if src.exists():
            continue
        # SU2 may write without .csv extension in some versions.
        alt = case_dir / name.replace(".csv", "")
        if alt.exists() and name.endswith(".csv"):
            shutil.copy2(alt, src)
