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
    RUN_NAME="${RUN_NAME:-synthetic_medium_000001_cluster}" \
    MATRIX_NAME="${MATRIX_NAME:-synthetic_medium_000001_cluster}" \
    SYMBOLS="${SYMBOLS:-000001}" \
    MODE="${MODE:-medium}" \
    DEVICE="${DEVICE:-cuda}" \
    DATA_DIR="${DATA_DIR:-/cluster/work/math/${USER}/mlfcs-gapa/data/${MATRIX_NAME:-synthetic_medium_000001_cluster}}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/project/math/${USER}/mlfcs-gapa/artifacts_sim}" \
    sbatch --parsable \
        ${ACCOUNT:+--account="${ACCOUNT}"} \
        cluster/euler_lobmmsim_matrix.sbatch
)
