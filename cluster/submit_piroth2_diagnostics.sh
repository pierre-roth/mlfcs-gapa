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
    RUN_NAME="${RUN_NAME:-piroth2_diagnostics}" \
    KIND="${KIND:-diagnostics}" \
    SYMBOL="${SYMBOL:-000001}" \
    MODE="${MODE:-medium}" \
    DEVICE="${DEVICE:-cpu}" \
    DATA_DIR="${DATA_DIR:-/cluster/work/math/${USER}/mlfcs-gapa/data/${RUN_NAME:-piroth2_diagnostics}}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/project/math/${USER}/mlfcs-gapa/artifacts_piroth2}" \
    sbatch --parsable \
        ${ACCOUNT:+--account="${ACCOUNT}"} \
        ${EXTRA_SBATCH_ARGS:-} \
        cluster/euler_piroth2_diagnostics.sbatch
)
