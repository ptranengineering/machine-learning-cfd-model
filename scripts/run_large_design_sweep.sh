#!/usr/bin/env bash
# Generate and run a large design-space CFD sweep (500+ cases) with resume support.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
N_SAMPLES="${N_SAMPLES:-500}"
SEED="${SEED:-1}"
RETRY_MAX="${RETRY_MAX:-5}"
SOLVER="${SOLVER:-rans}"

DESIGN_SPACE="${ROOT}/datasets/raw/design_space.csv"
RAW_OUTPUT="${ROOT}/datasets/raw/aero_design_raw.csv"
PROCESSED="${ROOT}/datasets/processed/aero_design_dataset.csv"
QUALITY_REPORT="${ROOT}/datasets/processed/aero_design_quality_report.csv"
LOG_DIR="${ROOT}/results/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/large_sweep_${STAMP}.log"

mkdir -p "${LOG_DIR}"

{
  echo "=== Large design sweep ==="
  echo "timestamp=${STAMP}"
  echo "n_samples=${N_SAMPLES} seed=${SEED} solver=${SOLVER} retry_max=${RETRY_MAX}"

  echo "--- Step 1: sample design space ---"
  "${PY}" "${ROOT}/scripts/generate_design_space.py" \
    --n-samples "${N_SAMPLES}" --seed "${SEED}"

  echo "--- Step 2: SU2 sweep (resume enabled) ---"
  "${PY}" "${ROOT}/scripts/run_design_sweep.py" \
    --solver "${SOLVER}" \
    --retry-max "${RETRY_MAX}" \
    --skip-completed \
    --append-raw-output \
    --design-space "${DESIGN_SPACE}" \
    --raw-output "${RAW_OUTPUT}"

  echo "--- Step 3: build dataset + quality report ---"
  "${PY}" "${ROOT}/scripts/build_dataset.py" \
    --dataset-type design \
    --design-raw "${RAW_OUTPUT}" \
    --design-output "${PROCESSED}" \
    --design-quality-report "${QUALITY_REPORT}"

  echo "--- Done ---"
} 2>&1 | tee "${LOG}"

echo "[DONE] log -> ${LOG}"
