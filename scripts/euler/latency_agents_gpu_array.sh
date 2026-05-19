#!/bin/bash
#SBATCH --job-name=mlfcs-latency
#SBATCH --account=ls_math
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --array=0-11%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
cd "${SCRATCH:?}/mlfcs-gapa"
source .venv/bin/activate

METHODS=("C-PPO" "D-DQN")
LATENCIES=(1 5 10 20 50 100)
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / ${#LATENCIES[@]}))
LATENCY_INDEX=$((SLURM_ARRAY_TASK_ID % ${#LATENCIES[@]}))
METHOD="${METHODS[$METHOD_INDEX]}"
LATENCY="${LATENCIES[$LATENCY_INDEX]}"

EVENTS="${EVENTS:-6000}"
EPISODE_EVENTS="${EPISODE_EVENTS:-2000}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000}"
SEED="${SEED:-101}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH}/mlfcs-gapa/runs/latency-agents-gpu/${SLURM_ARRAY_JOB_ID}}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-}"
NORMALIZE_ACTIONS="${NORMALIZE_ACTIONS:-true}"

normalize_actions_flag="--normalize-actions"
if [[ "${NORMALIZE_ACTIONS}" == "false" ]]; then
  normalize_actions_flag="--no-normalize-actions"
fi

echo "method=${METHOD}"
echo "latency=${LATENCY}"
echo "gpu_concurrency_cap=4"
echo "events=${EVENTS} episode_events=${EPISODE_EVENTS} timesteps=${TOTAL_TIMESTEPS} seed=${SEED}"
echo "normalize_actions=${NORMALIZE_ACTIONS}"

if [[ "${METHOD}" == "C-PPO" ]]; then
  cmd=(
    mlfcs-gapa train-synthetic-ppo
    --output-dir "${RUN_ROOT}/c_ppo/latency_${LATENCY}"
    --events "${EVENTS}"
    --episode-events "${EPISODE_EVENTS}"
    --latency-events "${LATENCY}"
    --total-timesteps "${TOTAL_TIMESTEPS}"
    --n-steps "${PPO_N_STEPS:-512}"
    --batch-size "${PPO_BATCH_SIZE:-128}"
    --n-epochs "${PPO_N_EPOCHS:-4}"
    --learning-rate "${PPO_LEARNING_RATE:-1e-4}"
    --gamma "${PPO_GAMMA:-0.99}"
    --gae-lambda "${PPO_GAE_LAMBDA:-0.95}"
    --clip-range "${PPO_CLIP_RANGE:-0.2}"
    --ent-coef "${PPO_ENT_COEF:-0.0}"
    --vf-coef "${PPO_VF_COEF:-0.5}"
    --max-grad-norm "${PPO_MAX_GRAD_NORM:-0.5}"
    "${normalize_actions_flag}"
    --device cuda
    --seed "${SEED}"
  )
  if [[ -n "${ENCODER_CHECKPOINT}" ]]; then
    cmd+=(--encoder-checkpoint "${ENCODER_CHECKPOINT}")
  fi
  "${cmd[@]}"
else
  cmd=(
    mlfcs-gapa train-synthetic-ddqn
    --output-dir "${RUN_ROOT}/d_dqn/latency_${LATENCY}"
    --events "${EVENTS}"
    --episode-events "${EPISODE_EVENTS}"
    --latency-events "${LATENCY}"
    --total-timesteps "${TOTAL_TIMESTEPS}"
    --learning-starts "${DDQN_LEARNING_STARTS:-1000}"
    --buffer-size "${DDQN_BUFFER_SIZE:-50000}"
    --batch-size "${DDQN_BATCH_SIZE:-128}"
    --target-update-interval "${DDQN_TARGET_UPDATE:-1000}"
    --device cuda
    --seed "${SEED}"
  )
  if [[ -n "${ENCODER_CHECKPOINT}" ]]; then
    cmd+=(--encoder-checkpoint "${ENCODER_CHECKPOINT}")
  fi
  "${cmd[@]}"
fi
