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
    RUN_NAME="${RUN_NAME:-anja_suite}" \
    SYMBOLS="${SYMBOLS:-000001}" \
    MODE="${MODE:-full}" \
    DEVICE="${DEVICE:-cuda}" \
    DATA_DIR="${DATA_DIR:-/cluster/scratch/${USER}/data/${RUN_NAME:-anja_suite}}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/scratch/${USER}/artifacts_anja}" \
    sbatch --parsable \
        ${ACCOUNT:+--account="${ACCOUNT}"} \
        cluster/euler_anja_suite.sbatch
)
