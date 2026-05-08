from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / 'results'
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
MODEL_DIR = RESULTS_DIR / 'models'
PLOTS_DIR = RESULTS_DIR / 'plots'

INPUT_PARAMS = ['thickness', 'camber', 'position', 'aoa', 'reynolds']
OUTPUT_PARAMS = ['Cd', 'Cl', 'Cm']
PARAM_RANGES = {
    'thickness': (0.05, 0.20),
    'camber': (0.0, 0.10),
    'position': (0.1, 0.5),
    'aoa': (-5.0, 15.0),
    'reynolds': (1e5, 1e7)
}
RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 15
HIDDEN_LAYERS = [128, 64, 32]
DROPOUT_RATE = 0.1
DEVICE = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
CHECKPOINT_PATH = MODEL_DIR / 'cfd_surrogate_best.pt'
METRICS_PATH = RESULTS_DIR / 'metrics.json'
TRAIN_HISTORY_PATH = PLOTS_DIR / 'training_history.png'
PREDICTIONS_PATH = PLOTS_DIR / 'predictions_vs_targets.png'
ERROR_DISTRIBUTION_PATH = PLOTS_DIR / 'error_distribution.png'
RESIDUALS_PATH = PLOTS_DIR / 'residuals.png'

for path in (RESULTS_DIR, DATA_DIR, RAW_DATA_DIR, MODEL_DIR, PLOTS_DIR):
    path.mkdir(parents=True, exist_ok=True)
