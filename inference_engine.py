"""
Batch inference and lightweight design exploration using the trained surrogate.

Typical usage after training::

    from pathlib import Path
    from inference_engine import CFDExplorer
    from config import CHECKPOINT_PATH

    explorer = CFDExplorer.from_checkpoint(CHECKPOINT_PATH, device="cpu")
    preds = explorer.predict(X_new)

Or with objects already in memory::

    explorer = CFDExplorer(model, scaler_X, scaler_y, device="cuda")
    designs, preds, scores = explorer.optimize(n_samples=5000, objective="max_cl_cd")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from config import DEVICE, INPUT_PARAMS, OUTPUT_PARAMS, PARAM_RANGES, RANDOM_STATE
from model import load_model, predict

Objective = Literal["max_cl_cd", "max_cl", "min_cd"]


class CFDExplorer:
    """Surrogate-driven predictions and random search over the configured box."""

    def __init__(self, model, scaler_X, scaler_y, device: Optional[str] = None) -> None:
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.device = device if device is not None else DEVICE

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path | str, device: Optional[str] = None) -> "CFDExplorer":
        path = Path(checkpoint_path)
        model, scaler_X, scaler_y, _meta = load_model(path, device=device or DEVICE)
        return cls(model, scaler_X, scaler_y, device=device or DEVICE)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return denormalized Cd/Cl/Cm predictions for physical-scale inputs X."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(INPUT_PARAMS):
            raise ValueError(
                f"Expected {len(INPUT_PARAMS)} input columns {INPUT_PARAMS}, got shape {X.shape}"
            )
        return predict(
            self.model,
            X,
            scaler_X=self.scaler_X,
            scaler_y=self.scaler_y,
            device=self.device,
        )

    def run_batch(self, X: np.ndarray) -> np.ndarray:
        """Alias for :meth:`predict` (full-batch inference)."""
        return self.predict(X)

    def optimize(
        self,
        n_samples: int = 10_000,
        objective: Objective = "max_cl_cd",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Random uniform search in ``PARAM_RANGES``, ranked by *objective*.

        Returns
        -------
        designs, preds, scores
            Sorted best-first (scores descending).
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        rng = rng or np.random.default_rng(RANDOM_STATE)

        names = INPUT_PARAMS
        lows = np.array([PARAM_RANGES[k][0] for k in names], dtype=np.float64)
        highs = np.array([PARAM_RANGES[k][1] for k in names], dtype=np.float64)
        u = rng.uniform(0.0, 1.0, size=(n_samples, len(names)))
        X = lows + u * (highs - lows)

        preds = self.predict(X)

        cd_idx = OUTPUT_PARAMS.index("Cd")
        cl_idx = OUTPUT_PARAMS.index("Cl")

        if objective == "max_cl_cd":
            eps = 1e-12
            scores = preds[:, cl_idx] / np.maximum(preds[:, cd_idx], eps)
        elif objective == "max_cl":
            scores = preds[:, cl_idx]
        elif objective == "min_cd":
            scores = -preds[:, cd_idx]
        else:
            raise ValueError(f"Unknown objective {objective!r}; use max_cl_cd, max_cl, or min_cd.")

        order = np.argsort(scores)[::-1]
        return X[order], preds[order], scores[order]
