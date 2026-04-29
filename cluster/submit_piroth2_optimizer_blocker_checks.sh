#!/bin/bash

set -euo pipefail

repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.."
    pwd
}

submit_stage() {
    local stage="$1"
    local dependency="${2:-}"
    local kind="${3:-}"

    local -a env_args=(
        "ACCOUNT=${ACCOUNT:-ls_math}"
        "RUN_NAME=${RUN_NAME}"
        "SYMBOL=${SYMBOL}"
        "MODE=full"
        "CREATE_PLOTS=false"
        "PRETRAIN_DEVICE=${PRETRAIN_DEVICE:-cuda}"
        "TRAIN_PPO_DEVICE=${TRAIN_PPO_DEVICE:-cuda}"
        "TRAIN_DQN_DEVICE=${TRAIN_DQN_DEVICE:-cuda}"
        "EVALUATE_DEVICE=${EVALUATE_DEVICE:-cpu}"
        "PRETRAIN_TIME=${PRETRAIN_TIME:-12:00:00}"
        "TRAIN_PPO_TIME=${TRAIN_PPO_TIME:-2-00:00:00}"
        "TRAIN_DQN_TIME=${TRAIN_DQN_TIME:-2-00:00:00}"
        "EVALUATE_TIME=${EVALUATE_TIME:-06:00:00}"
        "PRETRAIN_CPUS=${PRETRAIN_CPUS:-8}"
        "TRAIN_PPO_CPUS=${TRAIN_PPO_CPUS:-8}"
        "TRAIN_DQN_CPUS=${TRAIN_DQN_CPUS:-8}"
        "EVALUATE_CPUS=${EVALUATE_CPUS:-4}"
        "PRETRAIN_MEM_PER_CPU=${PRETRAIN_MEM_PER_CPU:-8G}"
        "TRAIN_PPO_MEM_PER_CPU=${TRAIN_PPO_MEM_PER_CPU:-12G}"
        "TRAIN_DQN_MEM_PER_CPU=${TRAIN_DQN_MEM_PER_CPU:-12G}"
        "EVALUATE_MEM_PER_CPU=${EVALUATE_MEM_PER_CPU:-8G}"
    )
    [[ -n "${dependency}" ]] && env_args+=("DEPENDENCY=${dependency}")
    [[ -n "${kind}" ]] && env_args+=("KIND=${kind}")
    env "${env_args[@]}" "${RUN_ENV[@]}" "${REPO}/cluster/submit_piroth2.sh" "${stage}" | tail -n 1
}

submit_shared_pretrain() {
    local group="$1"
    local symbol="$2"
    shift 2

    SYMBOL="${symbol}"
    RUN_NAME="${RUN_NAME_PREFIX:-piroth2}_${group}_pretrain_${symbol}_${STAMP}"
    RUN_ENV=("$@")
    local pretrain_id
    pretrain_id="$(submit_stage pretrain)"
    local checkpoint="${OUTPUT_ROOT:-/cluster/project/math/${USER}/mlfcs-gapa/artifacts_piroth2}/${RUN_NAME}/models/attnlob_pretrain.pt"
    printf '%s|%s\n' "${pretrain_id}" "${checkpoint}"
}

submit_pipeline() {
    local group="$1"
    local algo="$2"
    local dataset="$3"
    local symbol="$4"
    local pretrain_id="$5"
    local checkpoint="$6"
    shift 6

    SYMBOL="${symbol}"
    RUN_NAME="${RUN_NAME_PREFIX:-piroth2}_${group}_${algo}_${dataset}_${symbol}_${STAMP}"
    RUN_ENV=("CHECKPOINT=${checkpoint}" "$@")

    local train_stage eval_kind train_id eval_id baseline_id
    case "${algo}" in
        ppo) train_stage="train-ppo"; eval_kind="evaluate-ppo" ;;
        dqn) train_stage="train-dqn"; eval_kind="evaluate-dqn" ;;
        *) echo "Unknown algo: ${algo}" >&2; exit 1 ;;
    esac
    train_id="$(submit_stage "${train_stage}" "afterok:${pretrain_id}")"
    eval_id="$(submit_stage evaluate "afterok:${train_id}" "${eval_kind}")"
    baseline_id="$(submit_stage evaluate "" paper-baselines)"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "${group}" "${algo}" "${dataset}" "${symbol}" "${RUN_NAME}" "${pretrain_id}" "${train_id}" "${eval_id}" "${baseline_id}" "${checkpoint}"
}

synth_env() {
    local seed="$1"
    printf '%s\n' \
        "DATA_SOURCE=synthetic" \
        "SEED=${seed}" \
        "NUM_DAYS=20" \
        "TRAIN_DAYS=12" \
        "TEST_DAYS=8" \
        "EVENTS_PER_DAY_OVERRIDE=60000" \
        "EPISODE_LENGTH=2000" \
        "MAX_TRAIN_EPISODES_PER_DAY=12" \
        "MAX_EVAL_EPISODES_PER_DAY=12" \
        "ORDER_FLOW_MEMORY=0.35" \
        "VOLATILITY_CLUSTER_STRENGTH=0.45" \
        "VOLATILITY_CLUSTER_PERSISTENCE=0.992"
}

real_env() {
    local seed="$1"
    local stride="$2"
    printf '%s\n' \
        "DATA_SOURCE=real" \
        "REAL_DATA_ROOT=/cluster/work/math/piroth/mlfcs-gapa/data/processed" \
        "REAL_EVENT_STRIDE=${stride}" \
        "REAL_START_TIME=09:30:00" \
        "REAL_END_TIME=16:00:00" \
        "SEED=${seed}" \
        "NUM_DAYS=10" \
        "TRAIN_DAYS=6" \
        "TEST_DAYS=4" \
        "EVENTS_PER_DAY_OVERRIDE=60000" \
        "EPISODE_LENGTH=2000" \
        "MAX_TRAIN_EPISODES_PER_DAY=12" \
        "MAX_EVAL_EPISODES_PER_DAY=12"
}

REPO="$(repo_root)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
printf 'group,algo,dataset,symbol,run_name,pretrain_job,train_job,eval_job,baseline_job,checkpoint\n'

COMMON_PPO=(
    "TORCH_BATCH_SIZE=4096"
    "TORCH_LEARNING_RATE=0.00025"
    "PPO_EPOCHS=32"
    "PPO_ROLLOUTS_PER_EPOCH=192"
    "PPO_UPDATE_EPOCHS=8"
    "PPO_INITIAL_LOG_STD=-1.35"
    "PPO_INITIAL_SPREAD_BIAS=-0.70"
    "PPO_ENTROPY_COEF=0.008"
    "PPO_ENTROPY_COEF_FINAL=0.0001"
)
COMMON_DQN=(
    "TORCH_BATCH_SIZE=2048"
    "TORCH_LEARNING_RATE=0.00025"
    "TORCH_EPOCHS=16"
    "DQN_REPLAY_SIZE=500000"
    "DQN_MIN_REPLAY=4096"
    "DQN_UPDATE_INTERVAL=96"
    "DQN_TARGET_UPDATE_STEPS=1000"
    "DQN_EPSILON_START=0.55"
    "DQN_EPSILON_END=0.04"
    "DQN_EPSILON_DECAY=0.92"
)
BASE_REWARD=(
    "REWARD_MODE=hybrid"
    "REWARD_USE_DAMPENED_PNL=false"
    "REWARD_PNL_WEIGHT=1.0"
    "REWARD_SPREAD_PENALTY_SCALE=0"
    "REWARD_SPREAD_PENALTY_WEIGHT=0"
    "MAKER_REBATE_PER_SHARE=0"
)
SYNTH_U2_Z2=(
    "TRADE_UNIT_OVERRIDE=2"
    "REWARD_USE_TRADING_PNL=false"
    "REWARD_TRADING_PNL_WEIGHT=0"
    "REWARD_USE_INVENTORY_PENALTY=true"
    "REWARD_ZETA=0.000002"
    "REWARD_INVENTORY_PENALTY_WEIGHT=1.0"
)
REAL_Z1_U1=(
    "TRADE_UNIT_OVERRIDE=1"
    "REWARD_USE_TRADING_PNL=false"
    "REWARD_TRADING_PNL_WEIGHT=0"
    "REWARD_USE_INVENTORY_PENALTY=true"
    "REWARD_ZETA=0.000001"
    "REWARD_INVENTORY_PENALTY_WEIGHT=1.0"
)

mapfile -t SYNTH_ENV < <(synth_env 23)
IFS='|' read -r synth_pretrain synth_checkpoint < <(submit_shared_pretrain "optblock_synth_seed23" "000858" "${SYNTH_ENV[@]}" "${COMMON_PPO[@]}")
for label in enc01 enc00; do
    case "${label}" in
        enc01) LR_ENV=("TORCH_ENCODER_LEARNING_RATE_SCALE=0.1" "TORCH_BACKBONE_LEARNING_RATE_SCALE=0.5") ;;
        enc00) LR_ENV=("TORCH_ENCODER_LEARNING_RATE_SCALE=0.0" "TORCH_BACKBONE_LEARNING_RATE_SCALE=0.5") ;;
    esac
    submit_pipeline "optblock_synth_${label}_seed23" ppo synth "000858" "${synth_pretrain}" "${synth_checkpoint}" "${SYNTH_ENV[@]}" "${COMMON_PPO[@]}" "${BASE_REWARD[@]}" "${SYNTH_U2_Z2[@]}" "${LR_ENV[@]}"
done

mapfile -t REAL_ENV < <(real_env 23 250)
IFS='|' read -r real_pretrain real_checkpoint < <(submit_shared_pretrain "optblock_real_s250_seed23" "GOOGL" "${REAL_ENV[@]}" "${COMMON_DQN[@]}")
for label in enc01 enc00; do
    case "${label}" in
        enc01) LR_ENV=("TORCH_ENCODER_LEARNING_RATE_SCALE=0.1" "TORCH_BACKBONE_LEARNING_RATE_SCALE=0.5") ;;
        enc00) LR_ENV=("TORCH_ENCODER_LEARNING_RATE_SCALE=0.0" "TORCH_BACKBONE_LEARNING_RATE_SCALE=0.5") ;;
    esac
    submit_pipeline "optblock_real_z1_u1_s250_${label}_seed23" dqn real "GOOGL" "${real_pretrain}" "${real_checkpoint}" "${REAL_ENV[@]}" "${COMMON_DQN[@]}" "${BASE_REWARD[@]}" "${REAL_Z1_U1[@]}" "${LR_ENV[@]}"
done
