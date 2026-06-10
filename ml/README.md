"""
CFD SURROGATE MODEL - Neural Network Surrogate for Aerodynamic Design Optimization

Project Overview:
=================
This project develops a neural network surrogate model that predicts aerodynamic 
properties (lift, drag, moment) from airfoil geometry parameters and flow conditions.

The surrogate model targets faster-than-CFD inference for design exploration.
Accuracy depends on dataset size and CFD fidelity — see `results/design_baseline_metrics.json`
readiness gate (not guaranteed by default).

Key Features:
- Parametric airfoil generation and CFD simulation
- Neural network surrogate model development
- Evaluation plots and metrics reporting
- Inference time benchmarking
- Training pipeline for legacy Solidworks CFD exports (see root README for SU2 design-space path)

Directory Structure:
====================
cfd_surrogate/
├── config.py              # Configuration and paths
├── data_handler.py        # Data loading, preprocessing, normalization
├── model.py               # Neural network architecture and training
├── evaluation.py          # Metrics and visualization
├── train.py              # Main training script
├── README.md             # This file
├── data/
│   ├── raw/              # CFD outputs from Solidworks (your exports)
│   └── processed/        # Cleaned, normalized data
├── models/               # Trained model checkpoints
├── results/              # Plots, metrics, summary
└── logs/                 # Training logs


SETUP INSTRUCTIONS:
===================

1. Install Requirements:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install numpy pandas scikit-learn matplotlib scipy

   (Replace cu118 with your CUDA version, or use 'cpu' if no GPU)

2. Project Structure:
   The config.py file creates necessary directories automatically.
   Just run: python train.py --help

3. GPU Check:
   In Python:
   >>> import torch
   >>> print(torch.cuda.is_available())
   >>> print(torch.cuda.get_device_name(0))

4. Verify Installation:
   python -c "from config import *; print('Import successful')"


WORKFLOW:
=========

PHASE 1: DATA GENERATION (Weeks 1-2)
-------------------------------------

Step 1a: Generate sampling plan
  python train.py --generate-sweep-plan --n-samples 100
  
  This creates: data/raw/cfd_sweep_plan.csv
  Contains: 100 random airfoil configurations

Step 1b: Run CFD in Solidworks
  - Import cfd_sweep_plan.csv into Solidworks Flow Simulation
  - Set up your CFD parameters:
    * Domain: 20c × 10c (c = chord length)
    * Mesh: ~50k cells (refine near airfoil)
    * BC: Uniform freestream, symmetry sides
  - Run all 100 cases
  - Export results: Create CSV with columns:
    thickness, camber, position, aoa, reynolds, Cd, Cl, Cm
  - Save as: data/raw/cfd_data.csv

  Note: You can start with smaller dataset (50 cases) for testing


PHASE 2: MODEL TRAINING (Weeks 3-5)
------------------------------------

Step 2: Train surrogate model
  python train.py --data-csv data/raw/cfd_data.csv
  
  This will:
  - Load and preprocess data (70% train, 15% val, 15% test)
  - Train neural network (200 epochs with early stopping)
  - Generate all evaluation plots
  - Save model checkpoint and metadata
  - Print results summary

  Expected output:
  - models/cfd_surrogate_best.pt (trained model)
  - results/training_history.png
  - results/predictions_vs_targets.png
  - results/error_distribution.png
  - results/residuals.png
  - results/parameter_sensitivity.png
  - results/results_summary.txt


PHASE 3: VALIDATION & ANALYSIS (Weeks 6-7)
-------------------------------------------

The train.py script automatically generates:
- Test set predictions vs. ground truth
- Error distributions and residual plots
- Parameter sensitivity analysis
- Inference time benchmarking
- Complete metrics summary

Review these plots for your paper!


PHASE 4: WRITING (Weeks 8-9)
-----------------------------

Use the generated plots and metrics for your manuscript:

Paper Structure:
1. Abstract (summary of method and results)
2. Introduction (CFD limitations, surrogate modeling)
3. Methodology
   - Network architecture (include architecture string)
   - Training procedure (batch size, learning rate, epochs)
   - Validation approach
4. Results
   - Training curves (use training_history.png)
   - Test performance metrics (use results_summary.txt)
   - Predictions vs targets (use predictions_vs_targets.png)
   - Error analysis (use error_distribution.png)
   - Inference speedup (cite benchmark results)
5. Discussion
   - Strengths and limitations
   - Parameter sensitivity analysis
   - Comparison to other methods
6. Conclusion & Future Work


CFD DATA FORMAT:
================

Your CSV from Solidworks should have columns:

thickness    | camber    | position | aoa   | reynolds    | Cd      | Cl     | Cm
0.12         | 0.02      | 0.30     | 0.0   | 500000      | 0.0123  | 0.456  | -0.012
0.15         | 0.03      | 0.35     | 5.0   | 1000000     | 0.0234  | 0.789  | -0.034
...

Tips:
- thickness: Thickness as % chord (0.08-0.18 typical)
- camber: Max camber as % chord (0.00-0.05)
- position: Position of max camber, % chord (0.25-0.40)
- aoa: Angle of attack, degrees (-5 to +15)
- reynolds: Reynolds number (1e5 to 1e7)
- Cd, Cl, Cm: Force/moment coefficients from CFD

See PARAM_RANGES in config.py to adjust expected ranges.


CUSTOMIZATION:
===============

Edit config.py to customize:

1. Dataset parameters:
   PARAM_RANGES = {
       'thickness': (0.08, 0.18),
       'camber': (0.00, 0.05),
       ...
   }

2. Neural network architecture:
   HIDDEN_LAYERS = [128, 64, 32]  # Adjust layer sizes
   DROPOUT_RATE = 0.2              # Regularization

3. Training parameters:
   LEARNING_RATE = 1e-3
   NUM_EPOCHS = 200
   BATCH_SIZE = 32
   EARLY_STOPPING_PATIENCE = 20

4. Output variables:
   OUTPUT_PARAMS = ['Cd', 'Cl', 'Cm']  # Adjust as needed


INFERENCE (After Training):
============================

Load and use your trained model:

```python
from model import load_model, predict
import numpy as np
import pickle

# Load model and scalers
model = load_model('models/cfd_surrogate_best.pt')
with open('models/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

# Create new input (airfoil parameters)
new_airfoil = np.array([
    [0.12, 0.02, 0.30, 0.0, 500000],  # thickness, camber, pos, aoa, reynolds
])

# Get predictions
predictions = predict(model, new_airfoil, scalers['X'], scalers['y'])
print(f"Predicted Cd, Cl, Cm: {predictions[0]}")

# Compare to CFD (if you ran it separately)
# Time: <1ms for neural network vs ~10 seconds for CFD (50x speedup!)
```


TROUBLESHOOTING:
================

1. CUDA out of memory:
   - Reduce BATCH_SIZE in config.py
   - Or use --device cpu to run on CPU

2. Model not improving:
   - Check if data is normalized correctly (data_handler.py logs this)
   - Try different HIDDEN_LAYERS architecture
   - Increase number of CFD samples

3. High error on test set:
   - Need more CFD training data (try 150-200 samples)
   - Check CFD data quality (remove outliers)
   - Consider simpler problem (fewer parameters, narrower ranges)

4. Training is slow:
   - Use GPU: make sure torch is using CUDA
   - Verify with: torch.cuda.is_available() in Python
   - Reduce dataset size for testing

5. Import errors:
   - Make sure you're in the project directory
   - Run: python -m pip install torch numpy pandas scikit-learn


PUBLICATION CHECKLIST:
======================

Before submitting to conference/journal:

✓ Dataset: 100+ CFD cases, diverse parameter space
✓ Model: Trained and converged (validation loss plateaued)
✓ Metrics: R² > 0.95, MAE < 5% on test set
✓ Plots: All 5 evaluation plots in results/
✓ Validation: Compare NN predictions to held-out CFD cases
✓ Benchmark: Document speedup (50-100x typical)
✓ Writing: Clear methodology, honest limitations
✓ Code: Clean, commented, reproducible
✓ Advisor: Review before submission

Expected Results:
- Training Loss: Decreases from ~0.1 to < 0.001
- Validation R²: >0.90-0.95
- Test MAE: < 5% of output range
- Inference: 1-5 ms per sample (vs 10-100s for CFD)


NEXT STEPS AFTER TRAINING:
===========================

1. Analyze Results:
   - Read results_summary.txt
   - Review all plots
   - Identify where model performs best/worst

2. Improve (if needed):
   - More training data? (add more CFD cases)
   - Better features? (change parameter ranges)
   - Different architecture? (adjust hidden layers)

3. Write Paper:
   - Use plots and metrics from results/
   - Follow structure in "PHASE 4" section above
   - Target AIAA Student Technical Conference

4. Find Advisor:
   - Show them your manuscript
   - Get feedback
   - Revise

5. Submit:
   - AIAA deadline (usually May-June)
   - Or open-access journal (Frontiers, etc.)


REFERENCES FOR PAPER:
=====================

Key papers to cite:

Surrogate Modeling:
- Forrester et al., 2008: "Engineering Design via Surrogate Modelling"
- Queipo et al., 2005: "Surrogate-based analysis and optimization"

Neural Networks for CFD:
- Raissi et al., 2019: "Physics-informed neural networks"
- Han et al., 2020: "Solving Forward and Inverse Problems of the Nonlinear Schrödinger Equation"

Machine Learning for Aerodynamics:
- Chen et al., 2021: "Machine learning for aerodynamic design"
- Sekar et al., 2019: "FastSurferCNN"

Aerodynamic Theory:
- Abbott & Von Doenhoff, 1959: "Theory of wing sections"
- Anderson, 2011: "Fundamentals of aerodynamics"


TIMELINE REMINDER:
==================

For summer publication:
- Week 1-2:  Data generation (100 CFD cases)
- Week 3-5:  Model training and evaluation
- Week 6-7:  Validation and analysis
- Week 8-9:  Paper writing
- Week 10-12: Find advisor, revise, SUBMIT!

You've got this! Good luck with your research!
