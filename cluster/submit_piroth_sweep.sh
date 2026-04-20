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
    RUN_NAME="${RUN_NAME:-piroth_sweep}" \
    SWEEP_NAME="${SWEEP_NAME:-piroth_sweep}" \
    SYMBOLS="${SYMBOLS:-000001}" \
    MODE="${MODE:-medium}" \
    DEVICE="${DEVICE:-cuda}" \
    DATA_DIR="${DATA_DIR:-/cluster/work/math/${USER}/mlfcs-gapa/data/${RUN_NAME:-piroth_sweep}}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/project/math/${USER}/mlfcs-gapa/artifacts_piroth}" \
    sbatch --parsable \
        ${ACCOUNT:+--account="${ACCOUNT}"} \
        cluster/euler_piroth_sweep.sbatch
)
