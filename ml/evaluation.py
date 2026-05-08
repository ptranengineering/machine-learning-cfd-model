"""
Evaluation metrics and visualization functions
Generates publication-quality plots and performance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
from scipy import stats
from config import *

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate prediction metrics
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary with metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

    # R² score per output dimension with safe denominator
    ss_res = np.sum((y_true - y_pred)**2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    r2_mean = np.mean(r2)

    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'R2_per_output': r2.tolist() if hasattr(r2, 'tolist') else float(r2),
        'R2_mean': float(r2_mean)
    }


# ============================================================================
# PUBLICATION-QUALITY PLOTS
# ============================================================================

def plot_training_history(history, output_dir=RESULTS_DIR):
    """
    Plot training and validation loss over epochs
    
    Args:
        history: Dictionary from trainer.fit()
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.semilogy(epochs, history['train_loss'], 'o-', label='Training Loss', linewidth=2)
    ax1.semilogy(epochs, history['val_loss'], 's-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Metrics plot
    ax2.plot(epochs, history['val_mae'], 'o-', label='MAE', linewidth=2)
    ax2.plot(epochs, history['val_rmse'], 's-', label='RMSE', linewidth=2)
    ax2.plot(epochs, history['val_r2'], '^-', label='R² Score', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Metric Value', fontsize=12)
    ax2.set_title('Validation Metrics', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'training_history.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training history plot to {output_path}")
    plt.close()


def plot_predictions_vs_targets(y_true, y_pred, output_names, output_dir=RESULTS_DIR):
    """
    Plot predicted vs. target values for each output
    
    Args:
        y_true: Ground truth values (n_samples, n_outputs)
        y_pred: Predicted values (n_samples, n_outputs)
        output_names: List of output variable names
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    n_outputs = y_true.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(6*n_outputs, 5))
    
    if n_outputs == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, output_names)):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        # Scatter plot
        ax.scatter(y_t, y_p, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        ss_res = np.sum((y_t - y_p)**2)
        ss_tot = np.sum((y_t - np.mean(y_t))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        ax.set_xlabel(f'Target {name}', fontsize=11)
        ax.set_ylabel(f'Predicted {name}', fontsize=11)
        ax.set_title(f'{name} (R² = {r2:.4f})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'predictions_vs_targets.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved predictions plot to {output_path}")
    plt.close()


def plot_error_distribution(y_true, y_pred, output_names, output_dir=RESULTS_DIR):
    """
    Plot prediction error distributions
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        output_names: List of output variable names
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    n_outputs = y_true.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(6*n_outputs, 5))
    
    if n_outputs == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, output_names)):
        errors = y_pred[:, i] - y_true[:, i]
        
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean = {errors.mean():.2e}')
        ax.set_xlabel(f'Prediction Error in {name}', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Error Distribution: {name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'error_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved error distribution plot to {output_path}")
    plt.close()


def plot_residuals(y_true, y_pred, output_names, output_dir=RESULTS_DIR):
    """
    Plot residuals vs. predictions (residual plot)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        output_names: List of output variable names
        output_dir: Directory to save plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    n_outputs = y_true.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(6*n_outputs, 5))
    
    if n_outputs == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, output_names)):
        residuals = y_pred[:, i] - y_true[:, i]
        
        ax.scatter(y_pred[:, i], residuals, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        ax.axhline(0, color='r', linestyle='--', linewidth=2)
        
        ax.set_xlabel(f'Predicted {name}', fontsize=11)
        ax.set_ylabel('Residual', fontsize=11)
        ax.set_title(f'Residual Plot: {name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'residuals.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved residual plot to {output_path}")
    plt.close()


def plot_parameter_sensitivity(model, X, y_true, scaler_X, scaler_y, 
                              param_names, output_names, output_dir=RESULTS_DIR):
    """
    Plot sensitivity of outputs to each input parameter
    Shows how changing one parameter affects predictions
    
    Args:
        model: Trained model
        X: Full input dataset (unnormalized)
        y_true: True outputs
        scaler_X: Input scaler
        scaler_y: Output scaler
        param_names: Names of input parameters
        output_names: Names of output variables
        output_dir: Directory to save plot
    """
    from model import predict
    
    n_params = X.shape[1]
    n_outputs = len(output_names)
    fig, axes = plt.subplots(n_outputs, n_params, figsize=(4*n_params, 4*n_outputs))
    
    if n_outputs == 1:
        axes = axes.reshape(1, -1)
    if n_params == 1:
        axes = axes.reshape(-1, 1)
    
    # Use median input values as baseline
    X_baseline = np.median(X, axis=0, keepdims=True)
    
    for param_idx in range(n_params):
        # Vary this parameter, keep others at baseline
        param_values = np.linspace(X[:, param_idx].min(), X[:, param_idx].max(), 50)
        X_varied = np.repeat(X_baseline, len(param_values), axis=0)
        X_varied[:, param_idx] = param_values
        
        # Get predictions
        y_varied = predict(model, X_varied, scaler_X, scaler_y)
        
        for output_idx in range(n_outputs):
            ax = axes[output_idx, param_idx]
            
            ax.plot(param_values, y_varied[:, output_idx], 'b-', linewidth=2)
            ax.set_xlabel(param_names[param_idx], fontsize=10)
            if param_idx == 0:
                ax.set_ylabel(output_names[output_idx], fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if param_idx == 0:
                ax.set_title(output_names[output_idx], fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'parameter_sensitivity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved parameter sensitivity plot to {output_path}")
    plt.close()


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def generate_results_summary(history, y_test, y_pred_test, output_names, 
                            output_dir=RESULTS_DIR):
    """
    Generate a text summary of all results
    
    Args:
        history: Training history dictionary
        y_test: Test ground truth
        y_pred_test: Test predictions
        output_names: Names of outputs
        output_dir: Directory to save summary
    """
    summary = []
    summary.append("=" * 70)
    summary.append("CFD SURROGATE MODEL - RESULTS SUMMARY")
    summary.append("=" * 70)
    summary.append("")
    
    # Training summary
    summary.append("TRAINING SUMMARY:")
    summary.append(f"  Final Training Loss:      {history['train_loss'][-1]:.4e}")
    summary.append(f"  Final Validation Loss:    {history['val_loss'][-1]:.4e}")
    summary.append(f"  Best Validation Loss:     {min(history['val_loss']):.4e}")
    summary.append("")
    
    # Test metrics
    summary.append("TEST SET PERFORMANCE:")
    summary.append(f"  Test Loss (MSE):          {history['test_loss']:.4e}")
    summary.append(f"  Mean Absolute Error:      {history['test_mae']:.4e}")
    summary.append(f"  Root Mean Squared Error:  {history['test_rmse']:.4e}")
    summary.append(f"  R² Score:                 {history['test_r2']:.4f}")
    summary.append("")
    
    # Per-output metrics
    summary.append("PER-OUTPUT METRICS (Test Set):")
    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred_test)
    if y_test_arr.ndim == 1:
        y_test_arr = y_test_arr.reshape(-1, 1)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)
    r2_list = calculate_metrics(y_test, y_pred_test)["R2_per_output"]
    if not isinstance(r2_list, list):
        r2_list = [float(r2_list)]

    for i, name in enumerate(output_names):
        yt = y_test_arr[:, i]
        yp = y_pred_arr[:, i]
        mae_i = float(np.mean(np.abs(yp - yt)))
        rmse_i = float(np.sqrt(np.mean((yp - yt) ** 2)))
        r2_i = float(r2_list[i]) if i < len(r2_list) else float("nan")
        summary.append(f"\n  {name}:")
        summary.append(f"    MAE:  {mae_i:.4e}")
        summary.append(f"    RMSE: {rmse_i:.4e}")
        summary.append(f"    R²:   {r2_i:.4f}")
    
    summary.append("\n" + "=" * 70)
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    # Save to file
    output_path = Path(output_dir) / 'results_summary.txt'
    with open(output_path, 'w') as f:
        f.write(summary_text)
    
    print(f"\nSaved results summary to {output_path}")
