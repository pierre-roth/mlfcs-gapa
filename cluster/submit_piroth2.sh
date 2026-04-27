#!/bin/bash

set -euo pipefail

RESOURCE_FLAGS=()

usage() {
    cat <<'EOF'
Usage:
  cluster/submit_piroth2.sh diagnostics
  cluster/submit_piroth2.sh pretrain
  cluster/submit_piroth2.sh train-ppo
  cluster/submit_piroth2.sh train-dqn
  cluster/submit_piroth2.sh evaluate
  cluster/submit_piroth2.sh validate-data
  cluster/submit_piroth2.sh report
  cluster/submit_piroth2.sh suite
  cluster/submit_piroth2.sh pipeline

Environment:
  RUN_NAME=...              Shared output run name.
  SYMBOL=000001             Synthetic symbol.
  MODE=medium               smoke, medium, or full.
  DATA_DIR=/cluster/...     Defaults to /cluster/work/math/$USER/mlfcs-gapa/data/$RUN_NAME
  OUTPUT_ROOT=/cluster/...  Defaults to /cluster/project/math/$USER/mlfcs-gapa/artifacts_piroth2

Resource overrides:
  PRETRAIN_TIME=2-00:00:00
  TRAIN_PPO_GPUS=1
  TRAIN_DQN_MEM_PER_CPU=8G
  EVALUATE_PARTITION=normal.24h

Supported suffixes are TIME, CPUS, MEM_PER_CPU, GPUS, TMP, and PARTITION.
EOF
}

repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.."
    pwd
}

resource_flags() {
    local prefix="$1"
    local key value
    RESOURCE_FLAGS=()

    key="${prefix}_TIME"; value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--time="${value}")

    key="${prefix}_CPUS"; value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--cpus-per-task="${value}")

    key="${prefix}_MEM_PER_CPU"; value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--mem-per-cpu="${value}")

    key="${prefix}_GPUS"; value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--gpus="${value}")

    key="${prefix}_TMP"; value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--tmp="${value}")

    key="${prefix}_PARTITION"; value="${!key:-}"
    [[ -n "${value}" ]] && RESOURCE_FLAGS+=(--partition="${value}")

    return 0
}

submit_stage() {
    local stage_name="$1"
    local script_name="$2"
    local dependency="${3:-${DEPENDENCY:-}}"
    local kind="${4:-}"
    local device="${5:-}"
    local prefix

    prefix="$(printf '%s' "${stage_name}" | tr '[:lower:]-' '[:upper:]_')"
    resource_flags "${prefix}"

    local -a flags=(--parsable)
    if ((${#RESOURCE_FLAGS[@]})); then
        flags+=("${RESOURCE_FLAGS[@]}")
    fi
    [[ -n "${dependency}" ]] && flags+=(--dependency="${dependency}")
    [[ -n "${ACCOUNT:-}" ]] && flags+=(--account="${ACCOUNT}")

    local repo run_name
    repo="$(repo_root)"
    run_name="${RUN_NAME:-piroth2_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "${repo}/cluster/logs"

    (
        cd "${repo}"
        RUN_NAME="${run_name}" \
        KIND="${kind}" \
        SYMBOL="${SYMBOL:-000001}" \
        MODE="${MODE:-medium}" \
        DATA_DIR="${DATA_DIR:-/cluster/work/math/${USER}/mlfcs-gapa/data/${run_name}}" \
        OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/project/math/${USER}/mlfcs-gapa/artifacts_piroth2}" \
        DEVICE="${device}" \
        sbatch "${flags[@]}" "cluster/${script_name}"
    )
}

submit_pipeline() {
    local run_name="${RUN_NAME:-piroth2_$(date +%Y%m%d_%H%M%S)}"
    local pretrain_id ppo_id dqn_id ppo_eval_id dqn_eval_id baseline_id report_id dependency

    pretrain_id="$(RUN_NAME="${run_name}" submit_stage pretrain euler_piroth2_pretrain.sbatch "" "" "${PRETRAIN_DEVICE:-cuda}")"
    echo "Submitted pretrain: ${pretrain_id}"

    ppo_id="$(RUN_NAME="${run_name}" submit_stage train_ppo euler_piroth2_train_ppo.sbatch "afterok:${pretrain_id}" "" "${TRAIN_PPO_DEVICE:-cuda}")"
    echo "Submitted PPO train: ${ppo_id}"

    dqn_id="$(RUN_NAME="${run_name}" submit_stage train_dqn euler_piroth2_train_dqn.sbatch "afterok:${pretrain_id}" "" "${TRAIN_DQN_DEVICE:-cuda}")"
    echo "Submitted DQN train: ${dqn_id}"

    ppo_eval_id="$(RUN_NAME="${run_name}" submit_stage evaluate euler_piroth2_evaluate.sbatch "afterok:${ppo_id}" "evaluate-ppo" "${EVALUATE_DEVICE:-cpu}")"
    echo "Submitted PPO evaluation: ${ppo_eval_id}"

    dqn_eval_id="$(RUN_NAME="${run_name}" submit_stage evaluate euler_piroth2_evaluate.sbatch "afterok:${dqn_id}" "evaluate-dqn" "${EVALUATE_DEVICE:-cpu}")"
    echo "Submitted DQN evaluation: ${dqn_eval_id}"

    baseline_id="$(RUN_NAME="${run_name}" submit_stage evaluate euler_piroth2_evaluate.sbatch "" "paper-baselines" "${EVALUATE_DEVICE:-cpu}")"
    echo "Submitted baseline evaluation: ${baseline_id}"

    dependency="afterok:${ppo_eval_id}:${dqn_eval_id}:${baseline_id}"
    report_id="$(RUN_NAME="${run_name}" submit_stage report euler_piroth2_report.sbatch "${dependency}" "" "${REPORT_DEVICE:-cpu}")"
    echo "Submitted report: ${report_id}"
    echo "Run name: ${run_name}"
}

main() {
    if [[ $# -ne 1 ]]; then
        usage >&2
        exit 1
    fi

    case "$1" in
        diagnostics)
            submit_stage diagnostics euler_piroth2_diagnostics.sbatch "" diagnostics "${DIAGNOSTICS_DEVICE:-cpu}"
            ;;
        pretrain)
            submit_stage pretrain euler_piroth2_pretrain.sbatch "" "" "${PRETRAIN_DEVICE:-cuda}"
            ;;
        train-ppo)
            submit_stage train_ppo euler_piroth2_train_ppo.sbatch "" "" "${TRAIN_PPO_DEVICE:-cuda}"
            ;;
        train-dqn)
            submit_stage train_dqn euler_piroth2_train_dqn.sbatch "" "" "${TRAIN_DQN_DEVICE:-cuda}"
            ;;
        evaluate)
            submit_stage evaluate euler_piroth2_evaluate.sbatch "" "${KIND:-paper-baselines}" "${EVALUATE_DEVICE:-cpu}"
            ;;
        validate-data)
            submit_stage evaluate euler_piroth2_evaluate.sbatch "" synthetic-validation "${EVALUATE_DEVICE:-cpu}"
            ;;
        report)
            submit_stage report euler_piroth2_report.sbatch "" "" "${REPORT_DEVICE:-cpu}"
            ;;
        suite)
            submit_stage suite euler_piroth2_suite.sbatch "" "" "${SUITE_DEVICE:-cuda}"
            ;;
        pipeline)
            submit_pipeline
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
