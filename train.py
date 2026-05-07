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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CFD surrogate training, evaluation, and model export')
    parser.add_argument('--data', type=Path, default=RAW_DATA_DIR / 'cfd_data.csv', help='Path to CFD data CSV file')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_PATH, help='Path to save the trained model checkpoint')
    parser.add_argument('--metrics', type=Path, default=METRICS_PATH, help='Path to save evaluation metrics JSON')
    parser.add_argument('--plots-dir', type=Path, default=PLOTS_DIR, help='Directory to save plots')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Optimizer learning rate')
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE, help='Early stopping patience')
    parser.add_argument('--test-size', type=float, default=0.15, help='Fraction of data used for test set')
    parser.add_argument('--val-size', type=float, default=0.15, help='Fraction of remaining data used for validation')
    parser.add_argument('--device', type=str, default=DEVICE, choices=['cpu', 'cuda'], help='Device for training and inference')
    parser.add_argument('--generate-sweep-plan', action='store_true', help='Generate a CFD sampling plan CSV')
    parser.add_argument('--sweep-path', type=Path, default=RAW_DATA_DIR / 'cfd_sweep_plan.csv', help='Output path for generated sweep plan CSV')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of CFD samples to generate for sweep plan')
    return parser.parse_args()


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved metrics to {path}')


def save_predictions(X, y_true, y_pred, input_cols, output_cols, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(X, columns=input_cols)
    targets = pd.DataFrame(y_true, columns=[f'true_{name}' for name in output_cols])
    predictions = pd.DataFrame(y_pred, columns=[f'pred_{name}' for name in output_cols])
    result_df = pd.concat([data.reset_index(drop=True), targets, predictions], axis=1)
    result_df.to_csv(path, index=False)
    print(f'Saved predictions to {path}')


def main() -> None:
    args = parse_args()

    if args.generate_sweep_plan:
        export_cfd_sweep_plan(args.sweep_path, n_samples=args.n_samples)
        print(f'Generated CFD sweep plan: {args.sweep_path}')

    if args.data is None:
        if not args.generate_sweep_plan:
            print('Nothing to do. Use --data <path> to train or --generate-sweep-plan to create a sampling plan.')
        return

    data_path = args.data
    if not data_path.exists():
        if args.generate_sweep_plan:
            print(
                f'\nNo training data at {data_path} yet.\n'
                'Next steps:\n'
                '  1. Run your CFD batch using the sweep plan CSV.\n'
                '  2. Export results with columns: thickness,camber,position,aoa,reynolds,Cd,Cl,Cm\n'
                f'  3. Save as {data_path} (or pass --data <path>)\n'
                'Smoke-test without CFD:\n'
                '  python generate_demo_data.py\n'
                '  python train.py --data data/raw/cfd_demo_data.csv\n'
            )
            return
        raise FileNotFoundError(
            f'Input data file not found: {data_path}\n'
            'Please provide a valid CFD results CSV with --data or place the file at data/raw/cfd_data.csv'
        )

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but not available. Falling back to CPU.')
        device = 'cpu'

    pipeline = split_and_prepare_data(
        csv_filepath=data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        input_cols=INPUT_PARAMS,
        output_cols=OUTPUT_PARAMS,
        device=device
    )

    train_loader = pipeline['dataloaders']['train']
    val_loader = pipeline['dataloaders']['val']
    test_loader = pipeline['dataloaders']['test']
    X_test = pipeline['arrays']['X_test']
    y_test = pipeline['arrays']['y_test']
    X_test_norm = pipeline['normalized']['X_test']
    y_test_norm = pipeline['normalized']['y_test']
    X_scaler = pipeline['scalers']['X']
    y_scaler = pipeline['scalers']['y']
    metadata = pipeline['metadata']

    model = CFDSurrogateModel(
        n_inputs=metadata['n_inputs'],
        n_outputs=metadata['n_outputs']
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        device=device
    )

    trainer.fit(num_epochs=args.epochs, patience=args.patience)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(
        filepath=args.checkpoint,
        scaler_X=X_scaler,
        scaler_y=y_scaler,
        input_cols=metadata['input_cols'],
        output_cols=metadata['output_cols'],
        metadata={
            'random_state': RANDOM_STATE,
            'data_path': str(data_path)
        }
    )

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
        y_test_pred_norm = model(X_test_tensor).cpu().numpy()
    y_test_pred = y_scaler.inverse_transform(y_test_pred_norm)

    metrics = calculate_metrics(y_test, y_test_pred)
    metrics['training'] = {
        'epochs': len(trainer.history['train_loss']),
        'best_val_loss': min(trainer.history['val_loss']) if trainer.history['val_loss'] else None,
        'final_test_loss': float(trainer.history['test_loss']) if trainer.history['test_loss'] is not None else None,
        'final_test_r2': float(trainer.history['test_r2']) if trainer.history['test_r2'] is not None else None
    }

    save_metrics(metrics, args.metrics)
    save_predictions(X_test, y_test, y_test_pred, metadata['input_cols'], metadata['output_cols'], RESULTS_DIR / 'test_predictions.csv')

    plot_training_history(trainer.history, output_dir=args.plots_dir)
    plot_predictions_vs_targets(y_test, y_test_pred, metadata['output_cols'], output_dir=args.plots_dir)
    plot_error_distribution(y_test, y_test_pred, metadata['output_cols'], output_dir=args.plots_dir)
    plot_residuals(y_test, y_test_pred, metadata['output_cols'], output_dir=args.plots_dir)

    generate_results_summary(
        trainer.history,
        y_test,
        y_test_pred,
        metadata['output_cols'],
        output_dir=args.plots_dir,
    )

    print('\nReloading model checkpoint to validate restore...')
    loaded_model, loaded_X_scaler, loaded_y_scaler, checkpoint_meta = load_model(args.checkpoint, device=device)
    y_test_pred_reloaded = predict(loaded_model, X_test, scaler_X=loaded_X_scaler, scaler_y=loaded_y_scaler, device=device)
    max_diff = float(np.max(np.abs(y_test_pred_reloaded - y_test_pred)))
    print(f'Maximum difference between original and reloaded predictions: {max_diff:.6e}')
    print('Training and evaluation pipeline complete.')


if __name__ == '__main__':
    main()
