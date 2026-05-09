"""
Surrogate design optimizer web API + static UI.

Run from repository root:
    uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000

Requires trained model at results/models/design_rf_model.joblib (or MODEL_PATH).
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from optimize_design import (  # noqa: E402
    RANGE_PARAM_NAMES,
    default_bounds,
    run_surrogate_optimization,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_MODEL = ROOT / "results" / "models" / "design_rf_model.joblib"

_model_pack: dict[str, Any] | None = None
_model_error: str | None = None


def _model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL))).resolve()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model_pack, _model_error
    path = _model_path()
    try:
        if not path.is_file():
            _model_pack = None
            _model_error = f"Model file not found: {path}"
        else:
            _model_pack = joblib.load(path)
            _model_error = None
    except Exception as e:  # noqa: BLE001
        _model_pack = None
        _model_error = str(e)
    yield
    _model_pack = None


app = FastAPI(title="Aero surrogate optimizer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BoundsIn(BaseModel):
    geometry_thickness: tuple[float, float]
    geometry_camber: tuple[float, float]
    geometry_camber_pos: tuple[float, float]
    aoa: tuple[float, float]
    mach: tuple[float, float]
    reynolds: tuple[float, float]

    def to_lo_hi(self) -> tuple[Any, Any]:
        import numpy as np

        lo = []
        hi = []
        for name in RANGE_PARAM_NAMES:
            pair = getattr(self, name)
            lo.append(float(pair[0]))
            hi.append(float(pair[1]))
            if hi[-1] <= lo[-1]:
                raise ValueError(f"{name}: max must be greater than min")
        return np.array(lo, dtype=float), np.array(hi, dtype=float)


class OptimizeRequest(BaseModel):
    bounds: BoundsIn
    objective: Literal["max_cl_cd", "max_cl", "min_cd"] = "max_cl_cd"
    min_cl: float = Field(0.7, ge=-5.0, le=5.0)
    max_cd: float = Field(0.2, ge=1e-6, le=5.0)
    iters: int = Field(20, ge=1, le=200)
    init_samples: int = Field(25, ge=5, le=500)
    candidate_pool: int = Field(500, ge=50, le=5000)
    seed: int = Field(42, ge=0, le=2**31 - 1)


@app.get("/api/health")
def health() -> dict[str, Any]:
    path = _model_path()
    return {
        "ok": _model_error is None and _model_pack is not None,
        "model_path": str(path),
        "error": _model_error,
    }


@app.get("/api/defaults")
def api_defaults() -> dict[str, Any]:
    lo, hi = default_bounds()
    bounds = {name: [float(lo[i]), float(hi[i])] for i, name in enumerate(RANGE_PARAM_NAMES)}
    return {"bounds": bounds}


@app.post("/api/optimize")
def api_optimize(body: OptimizeRequest) -> dict[str, Any]:
    if _model_pack is None:
        raise HTTPException(
            status_code=503,
            detail=_model_error or "Model not loaded",
        )
    try:
        lo, hi = body.bounds.to_lo_hi()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        return run_surrogate_optimization(
            _model_pack,
            lo,
            hi,
            objective=body.objective,
            min_cl=body.min_cl,
            max_cd=body.max_cd,
            iters=body.iters,
            init_samples=body.init_samples,
            candidate_pool=body.candidate_pool,
            seed=body.seed,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
