"""
Data handling module for CFD Surrogate Model
Handles loading CFD outputs, preprocessing, and normalization
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import json
from config import *

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

class CFDDataset(Dataset):
    """PyTorch Dataset for CFD surrogate model"""
    
    def __init__(self, X, y):
        """
        Args:
            X: Input parameters (geometry + conditions)
            y: Output targets (aerodynamic properties)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_cfd_data(filepath):
    """
    Load CFD results from CSV exported from Solidworks
    
    Expected CSV format:
    thickness, camber, position, aoa, reynolds, Cd, Cl, Cm
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        pd.DataFrame with CFD data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} CFD samples from {filepath}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def preprocess_data(df, input_cols=None, output_cols=None):
    """
    Preprocess CFD data: handle missing values, remove outliers
    
    Args:
        df: DataFrame with CFD data
        input_cols: Column names for inputs (geometry params)
        output_cols: Column names for outputs (aerodynamic properties)
    
    Returns:
        Cleaned DataFrame
    """
    if input_cols is None:
        input_cols = ['thickness', 'camber', 'position', 'aoa', 'reynolds']
    if output_cols is None:
        output_cols = OUTPUT_PARAMS
    
    # Check all required columns exist
    required_cols = input_cols + output_cols
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    
    df = df[required_cols].copy()

    # Remove invalid rows FIRST
    df = df.dropna()

    # Safety check for physics validity
    if (df['reynolds'] <= 0).any():
        raise ValueError("Reynolds number must be positive for log transform")

    # Transform Reynolds (physics scaling)
    df['reynolds'] = np.log10(df['reynolds'])

    print(f"Rows after NaN removal: {len(df)}")

    # Clip INPUTS only (preserve CFD truth)
    for col in input_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].clip(mean - 3*std, mean + 3*std)

    print(f"Rows after input clipping: {len(df)}")
    
    return df


def normalize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Normalize input and output data using StandardScaler
    Fit on training data, apply to val/test
    
    Args:
        X_train, X_val, X_test: Input arrays
        y_train, y_val, y_test: Output arrays
    
    Returns:
        Tuple of (X_train_norm, X_val_norm, X_test_norm, 
                  y_train_norm, y_val_norm, y_test_norm,
                  X_scaler, y_scaler)
    """
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_norm = X_scaler.fit_transform(X_train)
    X_val_norm = X_scaler.transform(X_val)
    X_test_norm = X_scaler.transform(X_test)
    
    y_train_norm = y_scaler.fit_transform(y_train)
    y_val_norm = y_scaler.transform(y_val)
    y_test_norm = y_scaler.transform(y_test)
    
    print("Data normalized successfully")
    print(f"Input shape: {X_train_norm.shape}")
    print(f"Output shape: {y_train_norm.shape}")
    
    return (X_train_norm, X_val_norm, X_test_norm,
            y_train_norm, y_val_norm, y_test_norm,
            X_scaler, y_scaler)


def prepare_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, 
                       batch_size=BATCH_SIZE, device=DEVICE):
    """
    Create PyTorch DataLoaders for training/validation/testing
    
    Args:
        X_train, X_val, X_test: Normalized input arrays
        y_train, y_val, y_test: Normalized output arrays
        batch_size: Batch size for DataLoader
        device: 'cuda' or 'cpu' (used for pin_memory)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    pin_memory = True if device == 'cuda' else False
    train_dataset = CFDDataset(X_train, y_train)
    val_dataset = CFDDataset(X_val, y_val)
    test_dataset = CFDDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
    print(f"DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def split_and_prepare_data(csv_filepath, test_size=0.15, val_size=0.15,
                          input_cols=None, output_cols=None, device=DEVICE):
    """
    Complete data pipeline: load -> preprocess -> split -> normalize -> dataloaders
    
    Args:
        csv_filepath: Path to CFD data CSV
        test_size: Fraction for test set
        val_size: Fraction for validation set (of remaining after test split)
        input_cols: Input column names
        output_cols: Output column names
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary containing:
        - dataloaders (train, val, test)
        - scalers (X_scaler, y_scaler)
        - arrays (X_train, X_val, X_test, y_train, y_val, y_test)
        - normalized arrays (X_train_norm, etc.)
        - metadata (input_cols, output_cols, n_inputs, n_outputs)
    """
    if input_cols is None:
        input_cols = ['thickness', 'camber', 'position', 'aoa', 'reynolds']
    if output_cols is None:
        output_cols = OUTPUT_PARAMS
    
    # Load and preprocess
    df = load_cfd_data(csv_filepath)
    df = preprocess_data(df, input_cols, output_cols)
    
    # Extract inputs and outputs
    X = df[input_cols].values
    y = df[output_cols].values
    
    # Split: first test, then val from remainder
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Normalize
    (X_train_norm, X_val_norm, X_test_norm,
     y_train_norm, y_val_norm, y_test_norm,
     X_scaler, y_scaler) = normalize_data(X_train, X_val, X_test, 
                                          y_train, y_val, y_test)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        X_train_norm, X_val_norm, X_test_norm,
        y_train_norm, y_val_norm, y_test_norm,
        batch_size=BATCH_SIZE,
        device=device
    )
    
    return {
        'dataloaders': {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        },
        'scalers': {
            'X': X_scaler,
            'y': y_scaler
        },
        'arrays': {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        },
        'normalized': {
            'X_train': X_train_norm, 'X_val': X_val_norm, 'X_test': X_test_norm,
            'y_train': y_train_norm, 'y_val': y_val_norm, 'y_test': y_test_norm
        },
        'metadata': {
            'input_cols': input_cols,
            'output_cols': output_cols,
            'n_inputs': len(input_cols),
            'n_outputs': len(output_cols),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test)
        }
    }


# ============================================================================
# DATA EXPORT FOR CFD GENERATION
# ============================================================================

def generate_sampling_plan(n_samples=100, param_ranges=None):
    """
    Generate Latin Hypercube Sampling plan for CFD parameter sweep
    Ensures uniform coverage of parameter space
    
    Args:
        n_samples: Number of samples to generate
        param_ranges: Dict of parameter names and (min, max) tuples
    
    Returns:
        DataFrame with parameter combinations
    """
    from scipy.stats import qmc
    
    if param_ranges is None:
        param_ranges = PARAM_RANGES
    
    params = list(param_ranges.keys())
    bounds = [param_ranges[p] for p in params]
    
    # Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=len(params), random_state=42)
    samples = sampler.random(n_samples)
    samples_scaled = qmc.scale(samples, [b[0] for b in bounds], 
                                        [b[1] for b in bounds])
    
    df = pd.DataFrame(samples_scaled, columns=params)
    
    print(f"Generated {n_samples} parameter combinations")
    print(df.describe())
    
    return df


def export_cfd_sweep_plan(output_csv, n_samples=100):
    """
    Export parameter sweep plan as CSV for Solidworks CFD batch execution
    
    Args:
        output_csv: Path to save sweep plan
        n_samples: Number of CFD cases to run
    """
    df = generate_sampling_plan(n_samples=n_samples)
    df.to_csv(output_csv, index=False)
    print(f"Sweep plan exported to {output_csv}")
    print(f"Import this into Solidworks CFD for batch processing")
