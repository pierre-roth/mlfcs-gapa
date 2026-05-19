#!/bin/bash
#SBATCH --job-name=mlfcs-ablation
#SBATCH --account=ls_math
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --array=0-7%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
cd "${SCRATCH:?}/mlfcs-gapa"
source .venv/bin/activate

METHODS=("C-PPO" "D-DQN")
VARIANTS=("full" "without_lob" "without_attn_lob" "without_dynamic")
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / ${#VARIANTS[@]}))
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID % ${#VARIANTS[@]}))
METHOD="${METHODS[$METHOD_INDEX]}"
VARIANT="${VARIANTS[$VARIANT_INDEX]}"

EVENTS="${EVENTS:-6000}"
EPISODE_EVENTS="${EPISODE_EVENTS:-2000}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000}"
SEED="${SEED:-101}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH}/mlfcs-gapa/runs/ablation-agents-gpu/${SLURM_ARRAY_JOB_ID}}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-}"

LOB_MODE="attn"
USE_DYNAMIC_STATE="true"
USE_AGENT_STATE="true"
case "${VARIANT}" in
  full)
    ;;
  without_lob)
    LOB_MODE="none"
    ;;
  without_attn_lob)
    LOB_MODE="mlp"
    ;;
  without_dynamic)
    USE_DYNAMIC_STATE="false"
    ;;
  *)
    echo "unknown variant ${VARIANT}" >&2
    exit 2
    ;;
esac

dynamic_flag="--use-dynamic-state"
if [[ "${USE_DYNAMIC_STATE}" == "false" ]]; then
  dynamic_flag="--no-use-dynamic-state"
fi

echo "method=${METHOD}"
echo "variant=${VARIANT}"
echo "gpu_concurrency_cap=4"
echo "lob_mode=${LOB_MODE} use_dynamic_state=${USE_DYNAMIC_STATE}"
echo "events=${EVENTS} episode_events=${EPISODE_EVENTS} timesteps=${TOTAL_TIMESTEPS} seed=${SEED}"

if [[ "${METHOD}" == "C-PPO" ]]; then
  cmd=(
    mlfcs-gapa train-synthetic-ppo
    --output-dir "${RUN_ROOT}/c_ppo/${VARIANT}"
    --events "${EVENTS}"
    --episode-events "${EPISODE_EVENTS}"
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
    --lob-mode "${LOB_MODE}"
    "${dynamic_flag}"
    --use-agent-state
    --device cuda
    --seed "${SEED}"
  )
  if [[ -n "${ENCODER_CHECKPOINT}" && "${LOB_MODE}" == "attn" ]]; then
    cmd+=(--encoder-checkpoint "${ENCODER_CHECKPOINT}")
  fi
  "${cmd[@]}"
else
  cmd=(
    mlfcs-gapa train-synthetic-ddqn
    --output-dir "${RUN_ROOT}/d_dqn/${VARIANT}"
    --events "${EVENTS}"
    --episode-events "${EPISODE_EVENTS}"
    --total-timesteps "${TOTAL_TIMESTEPS}"
    --learning-starts "${DDQN_LEARNING_STARTS:-1000}"
    --buffer-size "${DDQN_BUFFER_SIZE:-50000}"
    --batch-size "${DDQN_BATCH_SIZE:-128}"
    --target-update-interval "${DDQN_TARGET_UPDATE:-1000}"
    --lob-mode "${LOB_MODE}"
    "${dynamic_flag}"
    --use-agent-state
    --device cuda
    --seed "${SEED}"
  )
  if [[ -n "${ENCODER_CHECKPOINT}" && "${LOB_MODE}" == "attn" ]]; then
    cmd+=(--encoder-checkpoint "${ENCODER_CHECKPOINT}")
  fi
  "${cmd[@]}"
fi
