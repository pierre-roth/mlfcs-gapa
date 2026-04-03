#!/bin/bash
# Validation job for lobmmx reward distortion fix.
# Runs pretrain + PPO (medium mode) on AAPL with Amine's scratch storage.
# Usage: bash cluster/submit_amine_validation.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="/cluster/scratch/abenjelloun"
DATA_DIR="${SCRATCH}/data/processed"
OUTPUT_ROOT="${SCRATCH}/artifacts"
RUN_NAME="amine_reward_fix_validation"

# --- pretrain ---
PRETRAIN_ID=$(sbatch --parsable \
  --job-name=lobmmx-pretrain \
  --account=public \
  --partition=normal \
  --time=04:00:00 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem-per-cpu=6G \
  --output="${REPO_ROOT}/cluster/logs/pretrain-%j.out" \
  --error="${REPO_ROOT}/cluster/logs/pretrain-%j.err" \
  --wrap="
    cd ${REPO_ROOT}
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=${REPO_ROOT}
    export DATA_DIR=${DATA_DIR}
    export OUTPUT_ROOT=${OUTPUT_ROOT}
    export RUN_NAME=${RUN_NAME}
    export SYMBOLS=AAPL
    export MODE=medium
    export DEVICE=cpu
    ~/.local/bin/uv run python cluster/euler_run_lobmmx.py pretrain
  ")
echo "Submitted pretrain: job ${PRETRAIN_ID}"

# --- PPO train (depends on pretrain) ---
TRAIN_ID=$(sbatch --parsable \
  --job-name=lobmmx-train \
  --account=public \
  --partition=normal \
  --time=08:00:00 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem-per-cpu=6G \
  --dependency="afterok:${PRETRAIN_ID}" \
  --output="${REPO_ROOT}/cluster/logs/train-%j.out" \
  --error="${REPO_ROOT}/cluster/logs/train-%j.err" \
  --wrap="
    cd ${REPO_ROOT}
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=${REPO_ROOT}
    export DATA_DIR=${DATA_DIR}
    export OUTPUT_ROOT=${OUTPUT_ROOT}
    export RUN_NAME=${RUN_NAME}
    export SYMBOLS=AAPL
    export MODE=medium
    export DEVICE=cpu
    export BACKBONE_RUN_NAME=${RUN_NAME}
    export REWARD_MODE=trade_inventory
    export RANDOM_INITIAL_INVENTORY=1
    export INITIAL_INVENTORY_MAX=125
    export ALLOW_TERMINAL_INVENTORY=1
    export TERMINAL_INVENTORY_COST_SCALE=1.0
    ~/.local/bin/uv run python cluster/euler_run_lobmmx.py train
  ")
echo "Submitted PPO train: job ${TRAIN_ID} (depends on ${PRETRAIN_ID})"
echo ""
echo "Monitor with: squeue -u abenjelloun"
echo "Logs: ${REPO_ROOT}/cluster/logs/"
