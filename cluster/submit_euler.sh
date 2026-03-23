#!/bin/bash

set -euo pipefail

RESOURCE_FLAGS=()

usage() {
    cat <<'EOF'
Usage:
  cluster/submit_euler.sh pretrain
  cluster/submit_euler.sh train
  cluster/submit_euler.sh evaluate
  cluster/submit_euler.sh report
  cluster/submit_euler.sh suite
  cluster/submit_euler.sh pipeline
  cluster/submit_euler.sh pipeline-aapl
  cluster/submit_euler.sh pipeline-aapl-medium

Environment overrides:
  RUN_NAME=...              Shared output run name. Recommended for multi-stage runs.
  SYMBOLS=AAPL,GOOGL        Comma-separated symbol list.
  DATA_DIR=/cluster/...     Defaults to /cluster/scratch/$USER/data/processed
  OUTPUT_ROOT=/cluster/...  Defaults to /cluster/scratch/$USER/artifacts
  MODE=full                 Defaults to full.

  PRETRAIN_TIME=2-00:00:00  Resource overrides per stage.
  TRAIN_TIME=2-00:00:00
  EVALUATE_TIME=2-00:00:00
  REPORT_TIME=2-00:00:00
  SUITE_TIME=2-00:00:00

  PRETRAIN_CPUS=8           Also supported: *_MEM_PER_CPU, *_GPUS, *_TMP, *_PARTITION.

Examples:
  RUN_NAME=euler_main SYMBOLS=AAPL,GOOGL cluster/submit_euler.sh pipeline
  RUN_NAME=euler_main cluster/submit_euler.sh pipeline-aapl
  RUN_NAME=euler_mid cluster/submit_euler.sh pipeline-aapl-medium
  RUN_NAME=euler_suite RUN_ABLATIONS=1 cluster/submit_euler.sh suite
EOF
}

repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.."
    pwd
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

symbols_array() {
    local raw="${SYMBOLS:-AAPL,GOOGL}"
    local part
    IFS=',' read -r -a parts <<<"${raw}"
    for part in "${parts[@]}"; do
        part="$(trim "${part}")"
        if [[ -n "${part}" ]]; then
            printf '%s\n' "${part}"
        fi
    done
}

resource_flags() {
    local prefix="$1"
    local key
    local value
    RESOURCE_FLAGS=()

    key="${prefix}_TIME"
    value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--time="${value}")

    key="${prefix}_CPUS"
    value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--cpus-per-task="${value}")

    key="${prefix}_MEM_PER_CPU"
    value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--mem-per-cpu="${value}")

    key="${prefix}_GPUS"
    value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--gpus="${value}")

    key="${prefix}_TMP"
    value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--tmp="${value}")

    key="${prefix}_PARTITION"
    value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--partition="${value}")
}

submit_stage() {
    local stage_name="$1"
    local script_name="$2"
    local dependency="${3:-}"
    local symbols_csv="${4:-${SYMBOLS:-AAPL,GOOGL}}"
    local device="${5:-}"

    local upper_stage
    upper_stage="$(printf '%s' "${stage_name}" | tr '[:lower:]' '[:upper:]')"

    resource_flags "${upper_stage}"
    local -a flags=(--parsable)
    if [[ "${RESOURCE_FLAGS+x}" == x ]]; then
        flags+=("${RESOURCE_FLAGS[@]+"${RESOURCE_FLAGS[@]}"}")
    fi
    if [[ -n "${dependency}" ]]; then
        flags+=(--dependency="${dependency}")
    fi

    if [[ -n "${ACCOUNT:-}" ]]; then
        flags+=(--account="${ACCOUNT}")
    fi

    local repo
    repo="$(repo_root)"
    mkdir -p "${repo}/cluster/logs"

    (
        cd "${repo}"
        RUN_NAME="${RUN_NAME:-}" \
        SYMBOLS="${symbols_csv}" \
        MODE="${MODE:-full}" \
        DATA_DIR="${DATA_DIR:-/cluster/scratch/${USER}/data/processed}" \
        OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/scratch/${USER}/artifacts}" \
        DEVICE="${device}" \
        sbatch "${flags[@]}" "cluster/${script_name}"
    )
}

submit_pipeline() {
    local run_name="${RUN_NAME:-euler_$(date +%Y%m%d_%H%M%S)}"
    local -a report_dependencies=()
    local symbol

    while IFS= read -r symbol; do
        local pretrain_id
        local train_id
        local evaluate_id

        pretrain_id="$(
            RUN_NAME="${run_name}" submit_stage pretrain euler_pretrain.sbatch "" "${symbol}" "${PRETRAIN_DEVICE:-cuda}"
        )"
        echo "Submitted pretrain for ${symbol}: ${pretrain_id}"

        train_id="$(
            RUN_NAME="${run_name}" submit_stage train euler_train_ppo.sbatch "afterok:${pretrain_id}" "${symbol}" "${TRAIN_DEVICE:-cuda}"
        )"
        echo "Submitted train for ${symbol}: ${train_id}"

        evaluate_id="$(
            RUN_NAME="${run_name}" submit_stage evaluate euler_evaluate.sbatch "afterok:${train_id}" "${symbol}" "${EVALUATE_DEVICE:-cpu}"
        )"
        echo "Submitted evaluate for ${symbol}: ${evaluate_id}"

        report_dependencies+=("${train_id}" "${evaluate_id}")
    done < <(symbols_array)

    local dependency
    dependency="afterok:$(IFS=:; echo "${report_dependencies[*]}")"
    local report_id
    report_id="$(
        RUN_NAME="${run_name}" submit_stage report euler_report.sbatch "${dependency}" "${SYMBOLS:-AAPL,GOOGL}" "${REPORT_DEVICE:-cpu}"
    )"
    echo "Submitted report: ${report_id}"
    echo "Run name: ${run_name}"
}

main() {
    if [[ $# -ne 1 ]]; then
        usage >&2
        exit 1
    fi

    case "$1" in
        pretrain)
            submit_stage pretrain euler_pretrain.sbatch "" "${SYMBOLS:-AAPL,GOOGL}" "${PRETRAIN_DEVICE:-cuda}"
            ;;
        train)
            submit_stage train euler_train_ppo.sbatch "" "${SYMBOLS:-AAPL,GOOGL}" "${TRAIN_DEVICE:-cuda}"
            ;;
        evaluate)
            submit_stage evaluate euler_evaluate.sbatch "" "${SYMBOLS:-AAPL,GOOGL}" "${EVALUATE_DEVICE:-cpu}"
            ;;
        report)
            submit_stage report euler_report.sbatch "" "${SYMBOLS:-AAPL,GOOGL}" "${REPORT_DEVICE:-cpu}"
            ;;
        suite)
            submit_stage suite euler_suite.sbatch "" "${SYMBOLS:-AAPL,GOOGL}" "${SUITE_DEVICE:-cuda}"
            ;;
        pipeline)
            submit_pipeline
            ;;
        pipeline-aapl)
            SYMBOLS=AAPL submit_pipeline
            ;;
        pipeline-aapl-medium)
            SYMBOLS=AAPL MODE=medium submit_pipeline
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac
}

main "$@"
