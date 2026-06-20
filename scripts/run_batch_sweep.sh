#!/usr/bin/env bash
# Run a small batch of new RANS cases, then rebuild the processed dataset.
# Safe to repeat: skips design_ids already successful in raw output.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
BATCH_SIZE="${BATCH_SIZE:-15}"
RETRY_MAX="${RETRY_MAX:-5}"
SOLVER="${SOLVER:-rans}"
DESIGN_N="${DESIGN_N:-500}"
SEED="${SEED:-1}"

DESIGN_SPACE="${ROOT}/datasets/raw/design_space.csv"
RAW_OUTPUT="${ROOT}/datasets/raw/aero_design_raw.csv"
PROCESSED="${ROOT}/datasets/processed/aero_design_dataset.csv"
QUALITY_REPORT="${ROOT}/datasets/processed/aero_design_quality_report.csv"

mkdir -p "${ROOT}/results/logs"

if [[ ! -f "${DESIGN_SPACE}" ]]; then
  echo "--- First run: sample design space (N=${DESIGN_N}) ---"
  "${PY}" "${ROOT}/scripts/generate_design_space.py" \
    --n-samples "${DESIGN_N}" --seed "${SEED}"
else
  echo "--- Reusing existing design space: ${DESIGN_SPACE} ---"
fi

echo "--- Batch sweep: up to ${BATCH_SIZE} new cases ---"
"${PY}" "${ROOT}/scripts/run_design_sweep.py" \
  --solver "${SOLVER}" \
  --retry-max "${RETRY_MAX}" \
  --skip-completed \
  --append-raw-output \
  --max-new "${BATCH_SIZE}" \
  --design-space "${DESIGN_SPACE}" \
  --raw-output "${RAW_OUTPUT}"

echo "--- Rebuild processed dataset ---"
"${PY}" "${ROOT}/scripts/build_dataset.py" \
  --dataset-type design \
  --design-raw "${RAW_OUTPUT}" \
  --design-output "${PROCESSED}" \
  --design-quality-report "${QUALITY_REPORT}"

"${PY}" -c "
import pandas as pd
raw = pd.read_csv('${RAW_OUTPUT}')
proc = pd.read_csv('${PROCESSED}')
ok = (raw['status'] == 'success').sum()
print(f'[SUMMARY] raw success={ok} / ${DESIGN_N}  |  processed accepted={len(proc)}')
"

echo "[DONE] batch session complete — re-run anytime with BATCH_SIZE=${BATCH_SIZE}"
