#!/bin/bash
#SBATCH --job-name=piroth2-thrscan
#SBATCH --account=ls_math
#SBATCH --partition=normal.24h
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=8G
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export REPO_ROOT
source "${REPO_ROOT}/cluster/euler_common.sh"

export DEVICE=cpu
export MODE=full
export RUN_NAME="${RUN_NAME:-piroth2_pretrain_threshold_scan}"
export DATA_DIR="${DATA_DIR:-/cluster/work/math/${USER}/mlfcs-gapa/data/processed}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/project/math/${USER}/mlfcs-gapa/artifacts_piroth2}"
mkdir -p "${OUTPUT_ROOT}"

euler_setup_job

python "${REPO_ROOT}/cluster/scan_piroth2_pretrain_thresholds.py" \
    --real-data-root "${REAL_DATA_ROOT:-/cluster/work/math/${USER}/mlfcs-gapa/data/processed}" \
    --symbols "${THRESHOLD_SCAN_SYMBOLS:-AAPL,GOOGL}" \
    --thresholds "${THRESHOLD_SCAN_VALUES:-0,0.0000025,0.000005,0.00001,0.00002,0.00005,0.0001}" \
    --horizon "${PRETRAIN_HORIZON:-10}" \
    --lookback "${LOOKBACK:-50}" \
    --train-days "${TRAIN_DAYS:-8}" \
    --test-days "${TEST_DAYS:-4}" \
    --start-time "${REAL_START_TIME:-09:30:00}" \
    --end-time "${REAL_END_TIME:-16:00:00}" \
    --output "${OUTPUT_ROOT}/pretrain_threshold_scan_${STAMP:-$(date +%Y%m%d_%H%M%S)}.csv"
