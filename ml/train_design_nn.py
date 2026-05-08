#!/usr/bin/env python3
"""
Train neural network on geometry + flow design dataset.
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
DEFAULT_DATA = ROOT / "datasets" / "processed" / "aero_design_dataset.csv"
DEFAULT_MODEL = ROOT / "results" / "models" / "design_nn_model.pt"
DEFAULT_METRICS = ROOT / "results" / "design_nn_metrics.json"


class DesignNet(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    p = argparse.ArgumentParser(description="Train design-space neural surrogate.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--epochs", type=int, default=350)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data)
    x_cols = ["geometry_param_1", "geometry_param_2", "geometry_param_3", "AoA", "Mach", "Re"]
    y_cols = ["CL", "CD"]
    x = df[x_cols].to_numpy(dtype=np.float32)
    y = df[y_cols].to_numpy(dtype=np.float32)
    if len(x) < 10:
        raise ValueError("Need at least 10 design samples for NN training.")

    x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, test_size=0.2, random_state=args.seed)
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.25, random_state=args.seed)

    sx, sy = StandardScaler(), StandardScaler()
    x_train_n = sx.fit_transform(x_train)
    x_val_n = sx.transform(x_val)
    x_test_n = sx.transform(x_test)
    y_train_n = sy.fit_transform(y_train)
    y_val_n = sy.transform(y_val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DesignNet(n_in=x_train_n.shape[1], n_out=y_train_n.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    tx = torch.tensor(x_train_n, dtype=torch.float32, device=device)
    ty = torch.tensor(y_train_n, dtype=torch.float32, device=device)
    vx = torch.tensor(x_val_n, dtype=torch.float32, device=device)
    vy = torch.tensor(y_val_n, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pred = model(tx)
        loss = loss_fn(pred, ty)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(vx), vy).item())

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if wait >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred_n = model(torch.tensor(x_test_n, dtype=torch.float32, device=device)).cpu().numpy()
    test_pred = sy.inverse_transform(test_pred_n)
    rmse = float(np.sqrt(np.mean((y_test - test_pred) ** 2)))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_cols": x_cols,
            "y_cols": y_cols,
            "x_scaler_mean": sx.mean_.tolist(),
            "x_scaler_scale": sx.scale_.tolist(),
            "y_scaler_mean": sy.mean_.tolist(),
            "y_scaler_scale": sy.scale_.tolist(),
        },
        args.model_out,
    )

    metrics = {
        "n_samples": int(len(df)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "test_rmse": rmse,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] metrics -> {args.metrics}")
    print(f"[DONE] model -> {args.model_out}")


if __name__ == "__main__":
    main()
