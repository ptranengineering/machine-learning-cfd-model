#!/usr/bin/env python3
"""
Physics-aware neural-network regressor for aerodynamic coefficients.

Uses multi-input features (as available) with normalization and
train/validation/test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "datasets" / "processed" / "aero_ml_dataset.csv"
DEFAULT_MODEL = ROOT / "results" / "models" / "nn_aero_regressor.pt"
DEFAULT_METRICS = ROOT / "results" / "nn_metrics.json"


class AeroRegressor(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_inputs
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_input_columns(df: pd.DataFrame, explicit: list[str] | None) -> list[str]:
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"Requested input columns not found: {missing}")
        return explicit

    preferred = [
        "aoa",
        "aoa_squared",
        "mach",
        "reynolds",
        "rms_rho_final",
        "rms_rho_u_final",
        "rms_rho_v_final",
        "rms_rho_e_final",
        "rms_rho_drop",
        "convergence_rate",
        "has_history",
        "thickness",
        "camber",
        "position",
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        raise ValueError("No usable numeric input columns found.")
    return cols


def split_three_way(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 samples for train/val/test split.")

    if n < 6:
        # Keep all three splits even for tiny data.
        test_n = 1
        val_n = 1
        train_n = n - test_n - val_n
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        train_idx = idx[:train_n]
        val_idx = idx[train_n : train_n + val_n]
        test_idx = idx[train_n + val_n :]
        return (
            x[train_idx],
            x[val_idx],
            x[test_idx],
            y[train_idx],
            y[val_idx],
            y[test_idx],
        )

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.25, random_state=seed
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def r2(a: np.ndarray, b: np.ndarray) -> float:
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a, axis=0)) ** 2))
    return 1.0 - (ss_res / (ss_tot + 1e-12))


def safe_r2(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    return r2(a, b)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train physics-aware NN for CL/CD prediction.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--inputs", type=str, default="", help="Comma-separated input column list.")
    p.add_argument("--outputs", type=str, default="cl,cd")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden", type=str, default="64,64,32")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data)
    in_cols = [c.strip() for c in args.inputs.split(",") if c.strip()] or None
    input_cols = select_input_columns(df, in_cols)
    output_cols = [c.strip() for c in args.outputs.split(",") if c.strip()]
    missing_outputs = [c for c in output_cols if c not in df.columns]
    if missing_outputs:
        raise ValueError(f"Missing output columns: {missing_outputs}")

    work_df = df[input_cols + output_cols].copy()
    work_df = work_df.apply(pd.to_numeric, errors="coerce").dropna()
    if len(work_df) < 3:
        raise ValueError("Not enough clean rows after numeric filtering.")

    x = work_df[input_cols].to_numpy(dtype=np.float32)
    y = work_df[output_cols].to_numpy(dtype=np.float32)
    x_train, x_val, x_test, y_train, y_val, y_test = split_three_way(x, y, args.seed)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_n = x_scaler.fit_transform(x_train)
    x_val_n = x_scaler.transform(x_val)
    x_test_n = x_scaler.transform(x_test)
    y_train_n = y_scaler.fit_transform(y_train)
    y_val_n = y_scaler.transform(y_val)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = AeroRegressor(
        n_inputs=len(input_cols),
        n_outputs=len(output_cols),
        hidden=[int(v) for v in args.hidden.split(",") if v.strip()],
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    tx = torch.tensor(x_train_n, dtype=torch.float32, device=device)
    ty = torch.tensor(y_train_n, dtype=torch.float32, device=device)
    vx = torch.tensor(x_val_n, dtype=torch.float32, device=device)
    vy = torch.tensor(y_val_n, dtype=torch.float32, device=device)

    best = {"val_loss": float("inf"), "state": None, "epoch": 0}
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(tx.shape[0], device=device)
        train_losses: list[float] = []
        for i in range(0, tx.shape[0], args.batch_size):
            idx = perm[i : i + args.batch_size]
            pred = model(tx[idx])
            loss = loss_fn(pred, ty[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(vx)
            val_loss = float(loss_fn(val_pred, vy).item())

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best["epoch"] = epoch
            wait = 0
        else:
            wait += 1

        if wait >= args.patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    model.eval()
    with torch.no_grad():
        ttest = torch.tensor(x_test_n, dtype=torch.float32, device=device)
        y_pred_n = model(ttest).cpu().numpy()
    y_pred = y_scaler.inverse_transform(y_pred_n)

    metrics = {
        "n_samples_total": int(len(work_df)),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "input_columns": input_cols,
        "output_columns": output_cols,
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "test_rmse": rmse(y_test, y_pred),
        "test_r2": safe_r2(y_test, y_pred),
        "per_output": {},
    }
    for i, name in enumerate(output_cols):
        metrics["per_output"][name] = {
            "rmse": rmse(y_test[:, i], y_pred[:, i]),
            "r2": safe_r2(y_test[:, i], y_pred[:, i]),
        }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "x_scaler_mean": x_scaler.mean_.tolist(),
        "x_scaler_scale": x_scaler.scale_.tolist(),
        "y_scaler_mean": y_scaler.mean_.tolist(),
        "y_scaler_scale": y_scaler.scale_.tolist(),
        "input_columns": input_cols,
        "output_columns": output_cols,
        "hidden": [int(v) for v in args.hidden.split(",") if v.strip()],
        "dropout": args.dropout,
    }
    torch.save(checkpoint, args.model_out)
    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] NN model saved to {args.model_out}")
    print(f"[DONE] Metrics saved to {args.metrics_out}")
    print(f"[INFO] Inputs: {input_cols}")
    test_r2_text = "n/a" if metrics["test_r2"] is None else f"{metrics['test_r2']:.4f}"
    print(f"[INFO] test_rmse={metrics['test_rmse']:.6f} test_r2={test_r2_text}")


if __name__ == "__main__":
    main()
