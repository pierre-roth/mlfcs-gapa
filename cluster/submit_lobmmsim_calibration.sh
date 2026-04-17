#!/bin/bash

set -euo pipefail

repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.."
    pwd
}

(
    cd "$(repo_root)"
    RUN_NAME="${RUN_NAME:-synthetic_acceptance_sweep_cluster}" \
    SWEEP_NAME="${SWEEP_NAME:-synthetic_acceptance_sweep_cluster}" \
    SYMBOLS="${SYMBOLS:-000001}" \
    MODE="${MODE:-medium}" \
    DEVICE="${DEVICE:-cpu}" \
    DATA_DIR="${DATA_DIR:-/cluster/work/math/${USER}/mlfcs-gapa/data/${SWEEP_NAME:-synthetic_acceptance_sweep_cluster}}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/project/math/${USER}/mlfcs-gapa/artifacts_sim}" \
    sbatch --parsable \
        ${ACCOUNT:+--account="${ACCOUNT}"} \
        cluster/euler_lobmmsim_calibration.sbatch
)
