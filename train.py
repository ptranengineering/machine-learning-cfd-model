import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_handler import export_cfd_sweep_plan, split_and_prepare_data
from evaluation import (
    calculate_metrics,
    generate_results_summary,
    plot_training_history,
    plot_predictions_vs_targets,
    plot_error_distribution,
    plot_residuals,
)
from model import CFDSurrogateModel, Trainer, load_model, predict
from config import (
    RAW_DATA_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    CHECKPOINT_PATH,
    METRICS_PATH,
    INPUT_PARAMS,
    OUTPUT_PARAMS,
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    RANDOM_STATE
)

# =========================================================
# REPRODUCIBILITY CORE
# =========================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# ARG PARSER (CLI STILL WORKS)
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="CFD Surrogate Training Pipeline")

    parser.add_argument("--data", type=Path, default=RAW_DATA_DIR / "cfd_data.csv")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)

    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)

    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)

    parser.add_argument("--device", type=str, default=DEVICE, choices=["cpu", "cuda"])

    parser.add_argument("--seed", type=int, default=RANDOM_STATE)

    parser.add_argument("--generate-sweep-plan", action="store_true")
    parser.add_argument("--sweep-path", type=Path, default=RAW_DATA_DIR / "cfd_sweep_plan.csv")
    parser.add_argument("--n-samples", type=int, default=100)

    return parser.parse_args()


# =========================================================
# SAVE HELPERS
# =========================================================
def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] metrics → {path}")


def save_predictions(X, y_true, y_pred, input_cols, output_cols, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    df_in = pd.DataFrame(X, columns=input_cols)
    df_true = pd.DataFrame(y_true, columns=[f"true_{c}" for c in output_cols])
    df_pred = pd.DataFrame(y_pred, columns=[f"pred_{c}" for c in output_cols])

    df = pd.concat([df_in, df_true, df_pred], axis=1)
    df.to_csv(path, index=False)

    print(f"[SAVE] predictions → {path}")


# =========================================================
# CORE TRAINING FUNCTION (USED BY run_experiment.py)
# =========================================================
def train_model(
    data_path,
    checkpoint=CHECKPOINT_PATH,
    metrics=METRICS_PATH,
    plots_dir=PLOTS_DIR,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    patience=EARLY_STOPPING_PATIENCE,
    test_size=0.15,
    val_size=0.15,
    device=DEVICE,
    seed=RANDOM_STATE,
):
    set_seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available → using CPU")
        device = "cpu"

    # -------------------------
    # DATA
    # -------------------------
    pipeline = split_and_prepare_data(
        csv_filepath=data_path,
        test_size=test_size,
        val_size=val_size,
        input_cols=INPUT_PARAMS,
        output_cols=OUTPUT_PARAMS,
        device=device
    )

    train_loader = pipeline["dataloaders"]["train"]
    val_loader = pipeline["dataloaders"]["val"]
    test_loader = pipeline["dataloaders"]["test"]

    X_test = pipeline["arrays"]["X_test"]
    y_test = pipeline["arrays"]["y_test"]
    X_test_norm = pipeline["normalized"]["X_test"]

    X_scaler = pipeline["scalers"]["X"]
    y_scaler = pipeline["scalers"]["y"]
    metadata = pipeline["metadata"]

    # -------------------------
    # MODEL
    # -------------------------
    model = CFDSurrogateModel(
        n_inputs=metadata["n_inputs"],
        n_outputs=metadata["n_outputs"]
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=learning_rate,
        device=device
    )

    # -------------------------
    # TRAIN
    # -------------------------
    trainer.fit(num_epochs=epochs, patience=patience)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    trainer.save_checkpoint(
        filepath=checkpoint,
        scaler_X=X_scaler,
        scaler_y=y_scaler,
        input_cols=metadata["input_cols"],
        output_cols=metadata["output_cols"],
        metadata={
            "seed": seed,
            "data_path": str(data_path)
        }
    )

    # -------------------------
    # PREDICTION
    # -------------------------
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
        y_pred_norm = model(X_tensor).cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_norm)

    # -------------------------
    # METRICS
    # -------------------------
    metrics_dict = calculate_metrics(y_test, y_pred)

    metrics_dict["training"] = {
        "epochs": len(trainer.history["train_loss"]),
        "best_val_loss": min(trainer.history["val_loss"]),
        "final_test_r2": float(trainer.history["test_r2"]),
    }

    save_metrics(metrics_dict, metrics)
    save_predictions(
        X_test, y_test, y_pred,
        metadata["input_cols"],
        metadata["output_cols"],
        RESULTS_DIR / "test_predictions.csv"
    )

    # -------------------------
    # PLOTS
    # -------------------------
    plot_training_history(trainer.history, output_dir=plots_dir)
    plot_predictions_vs_targets(y_test, y_pred, metadata["output_cols"], output_dir=plots_dir)
    plot_error_distribution(y_test, y_pred, metadata["output_cols"], output_dir=plots_dir)
    plot_residuals(y_test, y_pred, metadata["output_cols"], output_dir=plots_dir)

    generate_results_summary(
        trainer.history,
        y_test,
        y_pred,
        metadata["output_cols"],
        output_dir=plots_dir,
    )

    print("[DONE] Training complete")

    return model, X_scaler, y_scaler, metrics_dict


# =========================================================
# CLI WRAPPER (OPTIONAL)
# =========================================================
def main():
    args = parse_args()

    if args.generate_sweep_plan:
        export_cfd_sweep_plan(args.sweep_path, n_samples=args.n_samples)
        print(f"[DONE] sweep plan → {args.sweep_path}")
        return

    if not args.data.exists():
        raise FileNotFoundError(args.data)

    train_model(
        data_path=args.data,
        checkpoint=args.checkpoint,
        metrics=args.metrics,
        plots_dir=args.plots_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        test_size=args.test_size,
        val_size=args.val_size,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()