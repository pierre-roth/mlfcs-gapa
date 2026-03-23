#!/bin/bash

set -euo pipefail

euler_repo_root() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        printf '%s\n' "${SLURM_SUBMIT_DIR}"
        return
    fi
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.."
    pwd
}

euler_load_modules() {
    source /cluster/software/stacks/2025-06/setup-env.sh
    module purge
    module load "${STACK_MODULE:-stack/2024-06}" "${PYTHON_MODULE:-python_cuda/3.11.6}"
}

euler_activate_overlay_env() {
    local scratch_root="${SCRATCH:-/cluster/scratch/${USER}}"
    export VENV_DIR="${VENV_DIR:-${scratch_root}/venvs/mlfcs-gapa-euler-py311}"
    mkdir -p "$(dirname "${VENV_DIR}")"

    local lock_dir="${VENV_DIR}.lock"
    cleanup_lock() {
        rmdir "${lock_dir}" 2>/dev/null || true
    }
    while ! mkdir "${lock_dir}" 2>/dev/null; do
        echo "Waiting for overlay environment lock: ${lock_dir}" >&2
        sleep 2
    done
    trap cleanup_lock RETURN

    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        echo "Creating overlay virtualenv at ${VENV_DIR}" >&2
        python -m venv --system-site-packages "${VENV_DIR}"
    fi

    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"

    if ! python -c 'import pyrallis' >/dev/null 2>&1; then
        echo "Installing missing dependency pyrallis into ${VENV_DIR}" >&2
        python -m pip install --disable-pip-version-check --quiet pyrallis
    fi

    trap - RETURN
    cleanup_lock
}

euler_prepare_runtime_env() {
    export REPO_ROOT="${REPO_ROOT:-$(euler_repo_root)}"
    export DATA_DIR="${DATA_DIR:-/cluster/scratch/${USER}/data/processed}"
    export OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/scratch/${USER}/artifacts}"
    export MODE="${MODE:-full}"
    export PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

    local thread_count="${SLURM_CPUS_PER_TASK:-1}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${thread_count}}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${thread_count}}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${thread_count}}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${thread_count}}"

    local tmp_root="${TMPDIR:-/tmp}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-${tmp_root}/mpl-${SLURM_JOB_ID:-$$}}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${tmp_root}/xdg-${SLURM_JOB_ID:-$$}}"
    mkdir -p "${REPO_ROOT}/cluster/logs" "${OUTPUT_ROOT}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

    if [[ ! -d "${DATA_DIR}" ]]; then
        echo "DATA_DIR does not exist: ${DATA_DIR}" >&2
        exit 1
    fi
}

euler_verify_device() {
    if [[ "${DEVICE:-cpu}" != "cuda" ]]; then
        return
    fi

    python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Requested DEVICE=cuda but torch.cuda.is_available() is False")

print(f"CUDA available: {torch.cuda.get_device_name(0)}")
PY
}

euler_print_job_summary() {
    echo "Repository: ${REPO_ROOT}"
    echo "Data dir:   ${DATA_DIR}"
    echo "Output dir: ${OUTPUT_ROOT}"
    echo "Mode:       ${MODE}"
    echo "Run name:   ${RUN_NAME:-<auto>}"
    echo "Symbols:    ${SYMBOLS:-<config default>}"
    echo "Device:     ${DEVICE:-<config default>}"
    echo "Account:    ${SLURM_JOB_ACCOUNT:-${ACCOUNT:-unknown}}"
    echo "Node:       $(hostname)"
}

euler_setup_job() {
    REPO_ROOT="${REPO_ROOT:-$(euler_repo_root)}"
    export REPO_ROOT
    euler_load_modules
    euler_activate_overlay_env
    euler_prepare_runtime_env
    euler_verify_device
    euler_print_job_summary
}

euler_run_kind() {
    local kind="$1"
    shift || true

    cd "${REPO_ROOT}"
    srun python "${REPO_ROOT}/cluster/euler_run.py" "${kind}" "$@"
}
