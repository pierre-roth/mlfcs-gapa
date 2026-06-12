#!/bin/bash
#SBATCH --job-name=mlfcs-full-replication
#SBATCH --account=ls_math
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs the entire synthetic paper replication in one job:
# Table I (pretraining), Table II (overall), Figure 2 (latency),
# Table III (runtime), Table IV (ablation), Figure 3 (attention),
# and Figure 4 (decision trace).
#
# Usage (from $HOME/projects/mlfcs-gapa, after scripts/euler/setup_venv.sh):
#   sbatch scripts/euler/full_replication_gpu.sh
# Optional W&B tracking:
#   WANDB_ENABLED=true sbatch scripts/euler/full_replication_gpu.sh

set -euo pipefail

module load stack/2024-06 python/3.12.8
PROJECT_CODE_DIR="${PROJECT_CODE_DIR:-${HOME}/projects/mlfcs-gapa}"
cd "${PROJECT_CODE_DIR}"
if [[ ! -d .venv ]]; then
  echo "missing .venv — run scripts/euler/setup_venv.sh first" >&2
  exit 1
fi
source .venv/bin/activate
source scripts/euler/wandb_env.sh

RUN_ROOT="${RUN_ROOT:-/cluster/work/math/piroth/mlfcs-gapa/runs/full-replication/${SLURM_JOB_ID:-manual}}"

# Paper-faithful split and episode constants; the remaining knobs are the
# synthetic-data calibration documented in docs/replication_notes.md.
# - 10,000 events/day leaves ~7,600 stable-window events per stock/day, so the
#   train panel has ~76k events per stock and each test day yields 3 episodes.
# - 200,000 agent timesteps is ~2.6 passes over the train panel ("once or
#   more" in the paper), with PPO collecting rollouts from 8 parallel envs.
# - Conv-LOB's 1024-length windows are capped in-code at 20,000 events;
#   PRETRAIN_EVENTS=0 lets the other models use the full train panel.
STOCKS="${STOCKS:-000001,000858,002415}"
TRAIN_DAYS="${TRAIN_DAYS:-10}"
TEST_DAYS="${TEST_DAYS:-11}"
EVENTS_PER_DAY="${EVENTS_PER_DAY:-10000}"
EPISODE_EVENTS="${EPISODE_EVENTS:-2000}"
PRETRAIN_EVENTS="${PRETRAIN_EVENTS:-0}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-5}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
AGENT_TIMESTEPS="${AGENT_TIMESTEPS:-200000}"
TABULAR_EPISODES="${TABULAR_EPISODES:-50}"
LATENCIES="${LATENCIES:-1,5,10,20,50,100}"
RUNTIME_TRAIN_TIMESTEPS="${RUNTIME_TRAIN_TIMESTEPS:-64}"
PPO_N_ENVS="${PPO_N_ENVS:-8}"
SEED="${SEED:-101}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${RUN_ROOT}"
echo "run_root=${RUN_ROOT}"
echo "stocks=${STOCKS} train_days=${TRAIN_DAYS} test_days=${TEST_DAYS}"
echo "events_per_day=${EVENTS_PER_DAY} episode_events=${EPISODE_EVENTS}"
echo "pretrain: events=${PRETRAIN_EVENTS} epochs=${PRETRAIN_EPOCHS} batch=${PRETRAIN_BATCH_SIZE}"
echo "agents: timesteps=${AGENT_TIMESTEPS} tabular_episodes=${TABULAR_EPISODES} ppo_n_envs=${PPO_N_ENVS}"
echo "latencies=${LATENCIES} seed=${SEED} device=${DEVICE}"

mlfcs-gapa run-full-synthetic-replication \
  --output-dir "${RUN_ROOT}" \
  --stocks "${STOCKS}" \
  --train-days "${TRAIN_DAYS}" \
  --test-days "${TEST_DAYS}" \
  --events-per-day "${EVENTS_PER_DAY}" \
  --episode-events "${EPISODE_EVENTS}" \
  --pretrain-events "${PRETRAIN_EVENTS}" \
  --pretrain-epochs "${PRETRAIN_EPOCHS}" \
  --pretrain-batch-size "${PRETRAIN_BATCH_SIZE}" \
  --agent-timesteps "${AGENT_TIMESTEPS}" \
  --tabular-episodes "${TABULAR_EPISODES}" \
  --latency-values "${LATENCIES}" \
  --runtime-train-timesteps "${RUNTIME_TRAIN_TIMESTEPS}" \
  --ppo-n-envs "${PPO_N_ENVS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${WANDB_ARGS[@]}"

echo "full replication artifacts:"
find "${RUN_ROOT}" -maxdepth 2 -name "*.csv" -o -maxdepth 2 -name "*.png" | sort
