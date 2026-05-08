#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASE_ROOT="${REPO_ROOT}/su2_cases"

CFG_FILE="${CFG_FILE:-inv_NACA0012.cfg}"
AOA_START="${AOA_START:-0}"
AOA_END="${AOA_END:-10}"
AOA_STEP="${AOA_STEP:-2}"
AOA_LIST_OVERRIDE="${AOA_LIST_OVERRIDE:-}"

RAW_DATASET="${REPO_ROOT}/datasets/raw/aero_dataset.csv"
PROCESSED_DIR="${REPO_ROOT}/datasets/processed"

mkdir -p "${REPO_ROOT}/datasets/raw" "${PROCESSED_DIR}"
echo "AoA,CL,CD" > "${RAW_DATASET}"

extract_coeff_forces() {
    local key="$1"
    local file="$2"
    awk -F'[:|]' -v k="${key}" '$1 ~ ("^Total " k "$") {gsub(/[[:space:]]+/, "", $2); print $2; exit}' "${file}"
}

extract_coeff_output() {
    local key="$1"
    local file="$2"
    awk -F'[:|]' -v k="${key}" '$1 ~ ("^Total " k "$") {gsub(/[[:space:]]+/, "", $2); print $2; exit}' "${file}"
}

validate_row_is_numeric() {
    local aoa="$1"
    local cl="$2"
    local cd="$3"
    local numeric_re='^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$'
    if [[ ! "$cl" =~ $numeric_re || ! "$cd" =~ $numeric_re ]]; then
        echo "[ERROR] Non-numeric CL/CD for AoA=${aoa}: CL=${cl}, CD=${cd}"
        return 1
    fi
}

if [[ -n "${AOA_LIST_OVERRIDE}" ]]; then
    # shellcheck disable=SC2206
    AOA_LIST=(${AOA_LIST_OVERRIDE})
else
    AOA_LIST=()
    current="${AOA_START}"
    while awk "BEGIN { exit !(${current} <= ${AOA_END}) }"; do
        AOA_LIST+=("${current}")
        current="$(awk "BEGIN { print ${current} + ${AOA_STEP} }")"
    done
fi

for aoa in "${AOA_LIST[@]}"; do
    echo "[INFO] Running AoA=${aoa}"
    case_dir="${CASE_ROOT}/AoA_${aoa}"
    mkdir -p "${case_dir}"

    sed -i -E "s/^AOA=.*/AOA= ${aoa}/" "${CASE_ROOT}/${CFG_FILE}"

    output_file="${case_dir}/output_${aoa}.txt"
    if ! (cd "${CASE_ROOT}" && SU2_CFD "${CFG_FILE}" > "${output_file}" 2>&1); then
        echo "[WARN] SU2 failed at AoA=${aoa}; writing NA row."
        echo "${aoa},NA,NA" >> "${RAW_DATASET}"
        continue
    fi

    # Preserve run artifacts per case directory.
    [[ -f "${CASE_ROOT}/forces_breakdown.dat" ]] && cp -f "${CASE_ROOT}/forces_breakdown.dat" "${case_dir}/forces_breakdown.dat"
    [[ -f "${CASE_ROOT}/history.csv" ]] && cp -f "${CASE_ROOT}/history.csv" "${case_dir}/history.csv"
    [[ -f "${CASE_ROOT}/surface_flow.csv" ]] && cp -f "${CASE_ROOT}/surface_flow.csv" "${case_dir}/surface_flow.csv"

    cl=""
    cd=""
    if [[ -f "${case_dir}/forces_breakdown.dat" ]]; then
        cl="$(extract_coeff_forces "CL" "${case_dir}/forces_breakdown.dat")"
        cd="$(extract_coeff_forces "CD" "${case_dir}/forces_breakdown.dat")"
    fi
    if [[ -z "${cl}" || -z "${cd}" ]]; then
        cl="$(extract_coeff_output "CL" "${output_file}")"
        cd="$(extract_coeff_output "CD" "${output_file}")"
    fi

    if [[ -z "${cl}" || -z "${cd}" ]]; then
        echo "[WARN] Could not parse CL/CD at AoA=${aoa}; writing NA row."
        echo "${aoa},NA,NA" >> "${RAW_DATASET}"
        continue
    fi

    if ! validate_row_is_numeric "${aoa}" "${cl}" "${cd}"; then
        echo "${aoa},NA,NA" >> "${RAW_DATASET}"
        continue
    fi

    echo "${aoa},${cl},${cd}" >> "${RAW_DATASET}"
done

echo "[DONE] Sweep dataset written: ${RAW_DATASET}"
echo "[DONE] Case outputs stored under: ${CASE_ROOT}/AoA_*"
