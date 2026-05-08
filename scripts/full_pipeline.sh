#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/results/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/full_pipeline_${RUN_ID}.log"
META_FILE="${LOG_DIR}/full_pipeline_${RUN_ID}.json"

mkdir -p "${LOG_DIR}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
RUN_TRAINING="${RUN_TRAINING:-0}"

{
  echo "[INFO] run_id=${RUN_ID}"
  echo "[INFO] step=run_sweep"
  "${SCRIPT_DIR}/run_sweep.sh"

  echo "[INFO] step=build_dataset"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_dataset.py"

  if [[ "${RUN_TRAINING}" == "1" ]]; then
    echo "[INFO] step=train_baseline"
    "${PYTHON_BIN}" "${REPO_ROOT}/ml/train_baseline.py"
    echo "[INFO] step=train_nn"
    "${PYTHON_BIN}" "${REPO_ROOT}/ml/train_nn.py"
  else
    echo "[INFO] step=train_models skipped (RUN_TRAINING=${RUN_TRAINING})"
  fi
} | tee "${LOG_FILE}"

cat > "${META_FILE}" <<EOF
{
  "run_id": "${RUN_ID}",
  "log_file": "${LOG_FILE}",
  "raw_dataset": "${REPO_ROOT}/datasets/raw/aero_dataset.csv",
  "processed_dataset": "${REPO_ROOT}/datasets/processed/aero_ml_dataset.csv",
  "quality_report": "${REPO_ROOT}/datasets/processed/aero_ml_quality_report.csv",
  "baseline_metrics": "${REPO_ROOT}/results/baseline_metrics.json",
  "nn_metrics": "${REPO_ROOT}/results/nn_metrics.json",
  "run_training": "${RUN_TRAINING}"
}
EOF

echo "[DONE] Full pipeline complete"
echo "[DONE] Log: ${LOG_FILE}"
echo "[DONE] Metadata: ${META_FILE}"
