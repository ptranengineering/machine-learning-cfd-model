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
MACH_LIST_OVERRIDE="${MACH_LIST_OVERRIDE:-}"
REYNOLDS_LIST_OVERRIDE="${REYNOLDS_LIST_OVERRIDE:-}"
RETRY_MAX="${RETRY_MAX:-2}"

RAW_DATASET="${REPO_ROOT}/datasets/raw/aero_dataset.csv"
PROCESSED_DIR="${REPO_ROOT}/datasets/processed"
RAW_DIR="${REPO_ROOT}/datasets/raw"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${CASE_ROOT}/runs/${RUN_ID}"
RUN_CASES_DIR="${RUN_ROOT}/cases"
MANIFEST_CSV="${RAW_DIR}/sweep_manifest_${RUN_ID}.csv"
RUN_META_JSON="${RAW_DIR}/sweep_run_${RUN_ID}.json"

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}" "${RUN_CASES_DIR}"
echo "run_id,case_id,aoa,mach,reynolds,cl,cd,status,attempts,case_dir" > "${RAW_DATASET}"
echo "run_id,case_id,aoa,mach,reynolds,attempt,status,reason,case_dir,output_file" > "${MANIFEST_CSV}"

extract_coeff_forces() {
    local key="$1"
    local file="$2"
    awk -F'[:|]' -v k="${key}" '$1 ~ ("^Total " k "$") {gsub(/[[:space:]]+/, "", $2); print $2; exit}' "${file}"
}

extract_cfg_aoa() {
    local cfg="$1"
    awk -F'=' '/^AOA=/{gsub(/[[:space:]]+/, "", $2); print $2; exit}' "${cfg}"
}

extract_cfg_value() {
    local cfg="$1"
    local key="$2"
    awk -F'=' -v k="${key}" '$1 ~ ("^" k "$") {gsub(/[[:space:]]+/, "", $2); print $2; exit}' "${cfg}"
}

set_cfg_value() {
    local cfg="$1"
    local key="$2"
    local value="$3"
    if awk -F'=' -v k="${key}" '$1 ~ ("^[[:space:]]*" k "[[:space:]]*$") {found=1} END {exit !found}' "${cfg}"; then
        sed -i -E "s/^${key}\\s*=.*/${key}= ${value}/" "${cfg}"
    else
        echo "${key}= ${value}" >> "${cfg}"
    fi
}

dedupe_cfg_key() {
    local cfg="$1"
    local key="$2"
    awk -v k="${key}" '
        BEGIN {seen=0}
        {
            if ($0 ~ ("^[[:space:]]*" k "[[:space:]]*=")) {
                if (seen==0) {
                    print $0
                    seen=1
                }
            } else {
                print $0
            }
        }
    ' "${cfg}" > "${cfg}.tmp" && mv "${cfg}.tmp" "${cfg}"
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

CFG_PATH="${CASE_ROOT}/${CFG_FILE}"
ORIGINAL_AOA="$(extract_cfg_aoa "${CFG_PATH}")"
ORIGINAL_MACH="$(extract_cfg_value "${CFG_PATH}" "MACH_NUMBER")"
ORIGINAL_REYNOLDS="$(extract_cfg_value "${CFG_PATH}" "REYNOLDS_NUMBER")"
restore_cfg_aoa() {
    if [[ -n "${ORIGINAL_AOA:-}" ]]; then
        sed -i -E "s/^AOA=.*/AOA= ${ORIGINAL_AOA}/" "${CFG_PATH}"
    fi
    if [[ -n "${ORIGINAL_MACH:-}" ]]; then
        sed -i -E "s/^MACH_NUMBER\\s*=.*/MACH_NUMBER= ${ORIGINAL_MACH}/" "${CFG_PATH}"
    fi
    if [[ -n "${ORIGINAL_REYNOLDS:-}" ]]; then
        if rg -n "^REYNOLDS_NUMBER\\s*=" "${CFG_PATH}" > /dev/null 2>&1; then
            sed -i -E "s/^REYNOLDS_NUMBER\\s*=.*/REYNOLDS_NUMBER= ${ORIGINAL_REYNOLDS}/" "${CFG_PATH}"
        fi
    fi
}
trap restore_cfg_aoa EXIT

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

if [[ -n "${MACH_LIST_OVERRIDE}" ]]; then
    # shellcheck disable=SC2206
    MACH_LIST=(${MACH_LIST_OVERRIDE})
else
    MACH_DEFAULT="$(extract_cfg_value "${CFG_PATH}" "MACH_NUMBER")"
    MACH_LIST=("${MACH_DEFAULT}")
fi

if [[ -n "${REYNOLDS_LIST_OVERRIDE}" ]]; then
    # shellcheck disable=SC2206
    REYNOLDS_LIST=(${REYNOLDS_LIST_OVERRIDE})
else
    RE_DEFAULT="$(extract_cfg_value "${CFG_PATH}" "REYNOLDS_NUMBER")"
    if [[ -n "${RE_DEFAULT}" ]]; then
        REYNOLDS_LIST=("${RE_DEFAULT}")
    else
        # Keep a placeholder dimension for inviscid cases.
        REYNOLDS_LIST=("0")
    fi
fi

case_counter=0
passed_cases=0
failed_cases=0

for mach in "${MACH_LIST[@]}"; do
    for reynolds in "${REYNOLDS_LIST[@]}"; do
        for aoa in "${AOA_LIST[@]}"; do
            case_counter=$((case_counter + 1))
            case_id="$(printf "case_%04d" "${case_counter}")"
            case_dir="${RUN_CASES_DIR}/${case_id}"
            mkdir -p "${case_dir}"

            echo "[INFO] ${case_id} AoA=${aoa} Mach=${mach} Re=${reynolds}"
            set_cfg_value "${CFG_PATH}" "AOA" "${aoa}"
            set_cfg_value "${CFG_PATH}" "MACH_NUMBER" "${mach}"
            if [[ "${reynolds}" != "0" ]]; then
                set_cfg_value "${CFG_PATH}" "REYNOLDS_NUMBER" "${reynolds}"
            fi
            dedupe_cfg_key "${CFG_PATH}" "AOA"
            dedupe_cfg_key "${CFG_PATH}" "MACH_NUMBER"
            dedupe_cfg_key "${CFG_PATH}" "REYNOLDS_NUMBER"

            cl=""
            cd=""
            status="failed"
            fail_reason="solver_failed"
            attempts=0

            for ((attempt=1; attempt<=RETRY_MAX+1; attempt++)); do
                attempts="${attempt}"
                output_file="${case_dir}/output_attempt_${attempt}.txt"
                if ! (cd "${CASE_ROOT}" && SU2_CFD "${CFG_FILE}" > "${output_file}" 2>&1); then
                    echo "${RUN_ID},${case_id},${aoa},${mach},${reynolds},${attempt},retry,solver_failed,${case_dir},${output_file}" >> "${MANIFEST_CSV}"
                    continue
                fi

                [[ -f "${CASE_ROOT}/forces_breakdown.dat" ]] && cp -f "${CASE_ROOT}/forces_breakdown.dat" "${case_dir}/forces_breakdown.dat"
                [[ -f "${CASE_ROOT}/history.csv" ]] && cp -f "${CASE_ROOT}/history.csv" "${case_dir}/history.csv"
                [[ -f "${CASE_ROOT}/surface_flow.csv" ]] && cp -f "${CASE_ROOT}/surface_flow.csv" "${case_dir}/surface_flow.csv"
                cp -f "${CFG_PATH}" "${case_dir}/cfg_used.cfg"

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
                    echo "${RUN_ID},${case_id},${aoa},${mach},${reynolds},${attempt},retry,parse_failed,${case_dir},${output_file}" >> "${MANIFEST_CSV}"
                    continue
                fi

                if ! validate_row_is_numeric "${aoa}" "${cl}" "${cd}"; then
                    echo "${RUN_ID},${case_id},${aoa},${mach},${reynolds},${attempt},retry,non_numeric_coeff,${case_dir},${output_file}" >> "${MANIFEST_CSV}"
                    continue
                fi

                status="success"
                fail_reason=""
                echo "${RUN_ID},${case_id},${aoa},${mach},${reynolds},${attempt},success,,${case_dir},${output_file}" >> "${MANIFEST_CSV}"
                break
            done

            cat > "${case_dir}/case_meta.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "case_id": "${case_id}",
  "aoa": ${aoa},
  "mach": ${mach},
  "reynolds": ${reynolds},
  "status": "${status}",
  "attempts": ${attempts}
}
EOF

            if [[ "${status}" == "success" ]]; then
                passed_cases=$((passed_cases + 1))
                echo "${RUN_ID},${case_id},${aoa},${mach},${reynolds},${cl},${cd},success,${attempts},${case_dir}" >> "${RAW_DATASET}"
            else
                failed_cases=$((failed_cases + 1))
                echo "${RUN_ID},${case_id},${aoa},${mach},${reynolds},NA,NA,failed,${attempts},${case_dir}" >> "${RAW_DATASET}"
                echo "[WARN] ${case_id} failed after ${attempts} attempts (${fail_reason})."
            fi
        done
    done
done

cat > "${RUN_META_JSON}" <<EOF
{
  "run_id": "${RUN_ID}",
  "cfg_file": "${CFG_PATH}",
  "aoa_count": ${#AOA_LIST[@]},
  "mach_count": ${#MACH_LIST[@]},
  "reynolds_count": ${#REYNOLDS_LIST[@]},
  "total_cases": ${case_counter},
  "passed_cases": ${passed_cases},
  "failed_cases": ${failed_cases},
  "retry_max": ${RETRY_MAX},
  "run_cases_dir": "${RUN_CASES_DIR}",
  "raw_dataset": "${RAW_DATASET}",
  "manifest_csv": "${MANIFEST_CSV}"
}
EOF

echo "[DONE] Sweep dataset written: ${RAW_DATASET}"
echo "[DONE] Sweep manifest written: ${MANIFEST_CSV}"
echo "[DONE] Run metadata written: ${RUN_META_JSON}"
echo "[DONE] Case outputs stored under: ${RUN_CASES_DIR}"
