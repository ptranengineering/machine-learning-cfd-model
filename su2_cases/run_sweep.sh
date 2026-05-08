#!/bin/bash

set -euo pipefail

# Usage:
#   ./run_sweep.sh
#   AOA_LIST_OVERRIDE="0 2 4" ./run_sweep.sh
#   CFG_FILE=inv_NACA0012.cfg ./run_sweep.sh

CFG_FILE="${CFG_FILE:-inv_NACA0012.cfg}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_ROOT="${REPO_ROOT}/datasets/raw"
PROCESSED_ROOT="${REPO_ROOT}/datasets/processed"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RAW_ROOT}/su2_sweep_${RUN_ID}"
CASES_DIR="${RUN_DIR}/cases"
PROCESSED_CSV="${PROCESSED_ROOT}/aero_dataset.csv"
LOCAL_CSV="${SCRIPT_DIR}/aero_dataset.csv"

# Angles of attack for the sweep (default or override via env var).
if [[ -n "${AOA_LIST_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    AOA_LIST=(${AOA_LIST_OVERRIDE})
else
    AOA_LIST=(0 2 4 6 8 10)
fi

mkdir -p "${CASES_DIR}" "${PROCESSED_ROOT}"

# Output dataset for ML (local + canonical processed location).
echo "AoA,CL,CD" > "${LOCAL_CSV}"
echo "AoA,CL,CD" > "${PROCESSED_CSV}"

extract_coeff() {
    local key="$1"
    local file="$2"
    awk -F'[:|]' -v k="${key}" '$1 ~ ("^Total " k "$") {gsub(/[[:space:]]+/, "", $2); print $2; exit}' "${file}"
}

for aoa in "${AOA_LIST[@]}"; do
    echo "Running AoA = $aoa"

    # SU2 v8 config key is AOA, not ANGLE_OF_ATTACK.
    sed -i -E "s/^AOA=.*/AOA= ${aoa}/" "${SCRIPT_DIR}/${CFG_FILE}"

    # Run solver and keep full log per AoA.
    if ! (cd "${SCRIPT_DIR}" && SU2_CFD "${CFG_FILE}" > "output_${aoa}.txt" 2>&1); then
        echo "SU2 failed at AoA=${aoa}; writing NA row."
        echo "${aoa},NA,NA" >> "${LOCAL_CSV}"
        echo "${aoa},NA,NA" >> "${PROCESSED_CSV}"
        continue
    fi

    # Parse from forces breakdown using delimiters to avoid grabbing "CL|" / "CD|".
    CL=$(extract_coeff "CL" "${SCRIPT_DIR}/forces_breakdown.dat")
    CD=$(extract_coeff "CD" "${SCRIPT_DIR}/forces_breakdown.dat")

    if [[ -z "$CL" || -z "$CD" ]]; then
        echo "Could not parse CL/CD at AoA=${aoa}; writing NA row."
        CL="NA"
        CD="NA"
    fi

    echo "${aoa},${CL},${CD}" | tee -a "${LOCAL_CSV}" >> "${PROCESSED_CSV}"

    # Keep per-case artifacts for dataset traceability and ML provenance.
    CASE_DIR="${CASES_DIR}/AoA_${aoa}"
    mkdir -p "${CASE_DIR}"
    cp -f "${SCRIPT_DIR}/output_${aoa}.txt" "${CASE_DIR}/"
    cp -f "${SCRIPT_DIR}/forces_breakdown.dat" "${CASE_DIR}/forces_breakdown_AoA_${aoa}.dat"
    [[ -f "${SCRIPT_DIR}/history.csv" ]] && cp -f "${SCRIPT_DIR}/history.csv" "${CASE_DIR}/history_AoA_${aoa}.csv"
    [[ -f "${SCRIPT_DIR}/surface_flow.csv" ]] && cp -f "${SCRIPT_DIR}/surface_flow.csv" "${CASE_DIR}/surface_flow_AoA_${aoa}.csv"
done

cp -f "${LOCAL_CSV}" "${RUN_DIR}/aero_dataset.csv"

echo "Sweep complete."
echo "Local CSV: ${LOCAL_CSV}"
echo "Processed CSV: ${PROCESSED_CSV}"
echo "Run artifacts: ${RUN_DIR}"
