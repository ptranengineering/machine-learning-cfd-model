"""
Neural Network model for CFD surrogate
Feedforward network with configurable architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
from config import *

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class CFDSurrogateModel(nn.Module):
    """
    Feedforward neural network for CFD surrogate modeling
    Input: Geometry parameters + flow conditions
    Output: Aerodynamic coefficients (Cd, Cl, Cm)
    """
    
    def __init__(self, n_inputs, n_outputs, hidden_layers=None, dropout_rate=DROPOUT_RATE):
        """
        Args:
            n_inputs: Number of input parameters
            n_outputs: Number of output properties to predict
            hidden_layers: List of hidden layer sizes (default: [128, 64, 32])
            dropout_rate: Dropout probability for regularization
        """
        super(CFDSurrogateModel, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = HIDDEN_LAYERS
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Build network
        layers = []
        prev_size = n_inputs
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, n_outputs))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)
    
    def get_architecture_string(self):
        """Return a string describing the network architecture"""
        arch = f"Input({self.n_inputs})"
        for h in self.hidden_layers:
            arch += f" -> {h} (ReLU, Dropout={self.dropout_rate})"
        arch += f" -> Output({self.n_outputs})"
        return arch


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Training loop for CFD surrogate model"""
    
    def __init__(self, model, train_loader, val_loader, test_loader,
                 learning_rate=LEARNING_RATE, device=DEVICE):
        """
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            learning_rate: Initial learning rate
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_r2': [],
            'test_loss': None,
            'test_mae': None,
            'test_rmse': None,
            'test_r2': None
        }
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Evaluate on validation set"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                val_loss += loss.item()
                
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        avg_val_loss = val_loss / len(self.val_loader)
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
        ss_res = np.sum((all_preds - all_targets)**2)
        ss_tot = np.sum((all_targets - all_targets.mean())**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        return avg_val_loss, mae, rmse, float(r2)
    
    def test(self):
        """Evaluate on test set"""
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                test_loss += loss.item()
                
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        avg_test_loss = test_loss / len(self.test_loader)
        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
        ss_res = np.sum((all_preds - all_targets)**2)
        ss_tot = np.sum((all_targets - all_targets.mean())**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        return avg_test_loss, mae, rmse, float(r2), all_preds, all_targets
    
    def fit(self, num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE):
        """
        Train the model with early stopping
        
        Args:
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
        
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model architecture: {self.model.get_architecture_string()}\n")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_mae, val_rmse, val_r2 = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_rmse'].append(val_rmse)
            self.history['val_r2'].append(val_r2)
            
            self.scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4e}, "
                      f"Val Loss: {val_loss:.4e}, "
                      f"Val MAE: {val_mae:.4e}, "
                      f"Val R²: {val_r2:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print("\nTraining complete. Running final test evaluation...")
        test_loss, test_mae, test_rmse, test_r2, test_preds, test_targets = self.test()
        
        self.history['test_loss'] = test_loss
        self.history['test_mae'] = test_mae
        self.history['test_rmse'] = test_rmse
        self.history['test_r2'] = test_r2
        self.test_preds = test_preds
        self.test_targets = test_targets
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4e}")
        print(f"  MAE:  {test_mae:.4e}")
        print(f"  RMSE: {test_rmse:.4e}")
        print(f"  R²:   {test_r2:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filepath=None, scaler_X=None, scaler_y=None,
                        input_cols=None, output_cols=None, metadata=None,
                        is_best=False):
        """Save model checkpoint with scalers and metadata"""
        if filepath is None:
            filepath = Path(CHECKPOINT_PATH)
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_inputs': self.model.n_inputs,
            'n_outputs': self.model.n_outputs,
            'hidden_layers': self.model.hidden_layers,
            'dropout_rate': self.model.dropout_rate,
            'history': self.history,
            'metadata': metadata or {},
            'input_cols': input_cols,
            'output_cols': output_cols
        }
        
        if scaler_X is not None:
            checkpoint['scaler_X'] = {
                'mean': scaler_X.mean_.tolist(),
                'scale': scaler_X.scale_.tolist()
            }
        if scaler_y is not None:
            checkpoint['scaler_y'] = {
                'mean': scaler_y.mean_.tolist(),
                'scale': scaler_y.scale_.tolist()
            }
        
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"Saved best model checkpoint to {filepath}")
        else:
            print(f"Saved model checkpoint to {filepath}")


# ============================================================================
# MODEL INFERENCE
# ============================================================================

def load_model(checkpoint_path, device=DEVICE):
    """
    Load a trained model and scalers from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: 'cuda' or 'cpu'
    
    Returns:
        model, scaler_X, scaler_y, checkpoint metadata
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = CFDSurrogateModel(
        n_inputs=checkpoint['n_inputs'],
        n_outputs=checkpoint['n_outputs'],
        hidden_layers=checkpoint['hidden_layers'],
        dropout_rate=checkpoint['dropout_rate']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_X = None
    scaler_y = None
    if 'scaler_X' in checkpoint:
        scaler_X = StandardScaler()
        scaler_X.mean_ = np.array(checkpoint['scaler_X']['mean'], dtype=np.float64)
        scaler_X.scale_ = np.array(checkpoint['scaler_X']['scale'], dtype=np.float64)
        scaler_X.var_ = scaler_X.scale_ ** 2
        scaler_X.n_features_in_ = len(scaler_X.mean_)
    if 'scaler_y' in checkpoint:
        scaler_y = StandardScaler()
        scaler_y.mean_ = np.array(checkpoint['scaler_y']['mean'], dtype=np.float64)
        scaler_y.scale_ = np.array(checkpoint['scaler_y']['scale'], dtype=np.float64)
        scaler_y.var_ = scaler_y.scale_ ** 2
        scaler_y.n_features_in_ = len(scaler_y.mean_)
    
    metadata = checkpoint.get('metadata', {})
    metadata['input_cols'] = checkpoint.get('input_cols')
    metadata['output_cols'] = checkpoint.get('output_cols')
    
    print(f"Loaded model from {checkpoint_path}")
    return model, scaler_X, scaler_y, metadata


def predict(model, X, scaler_X=None, scaler_y=None, device=DEVICE):
    """
    Make predictions on new data
    
    Args:
        model: Trained PyTorch model
        X: Input array (not normalized)
        scaler_X: StandardScaler for inputs
        scaler_y: StandardScaler for outputs
        device: 'cuda' or 'cpu'
    
    Returns:
        Predicted output values (denormalized)
    """
    model.eval()
    
    # Normalize inputs if scaler provided
    if scaler_X is not None:
        X_norm = scaler_X.transform(X)
    else:
        X_norm = X
    
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred_norm = model(X_tensor).cpu().numpy()
    
    # Denormalize outputs if scaler provided
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_norm)
    else:
        y_pred = y_pred_norm
    
    return y_pred


def inference_time_benchmark(model, X, n_runs=100, device=DEVICE):
    """
    Benchmark model inference time
    
    Args:
        model: Trained model
        X: Input data
        n_runs: Number of inference runs for averaging
        device: 'cuda' or 'cpu'
    
    Returns:
        Average time per inference
    """
    import time
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(X_tensor)
    
    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(X_tensor)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    
    avg_time = (end - start) / n_runs
    
    print(f"Inference time benchmark ({n_runs} runs):")
    print(f"  Average: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {len(X)/avg_time:.0f} samples/sec")
    
    return avg_time
