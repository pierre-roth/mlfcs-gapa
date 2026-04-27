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
        "PRETRAIN_TIME=${PRETRAIN_TIME:-10:00:00}"
        "TRAIN_PPO_TIME=${TRAIN_PPO_TIME:-1-12:00:00}"
        "TRAIN_DQN_TIME=${TRAIN_DQN_TIME:-1-12:00:00}"
        "EVALUATE_TIME=${EVALUATE_TIME:-04:00:00}"
        "PRETRAIN_CPUS=${PRETRAIN_CPUS:-8}"
        "TRAIN_PPO_CPUS=${TRAIN_PPO_CPUS:-8}"
        "TRAIN_DQN_CPUS=${TRAIN_DQN_CPUS:-8}"
        "EVALUATE_CPUS=${EVALUATE_CPUS:-4}"
        "PRETRAIN_MEM_PER_CPU=${PRETRAIN_MEM_PER_CPU:-8G}"
        "TRAIN_PPO_MEM_PER_CPU=${TRAIN_PPO_MEM_PER_CPU:-10G}"
        "TRAIN_DQN_MEM_PER_CPU=${TRAIN_DQN_MEM_PER_CPU:-10G}"
        "EVALUATE_MEM_PER_CPU=${EVALUATE_MEM_PER_CPU:-8G}"
    )
    if [[ -n "${dependency}" ]]; then
        env_args+=("DEPENDENCY=${dependency}")
    fi
    if [[ -n "${kind}" ]]; then
        env_args+=("KIND=${kind}")
    fi
    env "${env_args[@]}" "${RUN_ENV[@]}" "${REPO}/cluster/submit_piroth2.sh" "${stage}" | tail -n 1
}

submit_pipeline() {
    local group="$1"
    local algo="$2"
    local dataset="$3"
    local symbol="$4"
    shift 4

    SYMBOL="${symbol}"
    RUN_NAME="${RUN_NAME_PREFIX:-piroth2}_${group}_${algo}_${dataset}_${symbol}_${STAMP}"
    RUN_ENV=("$@")

    local pretrain_id train_id eval_id baseline_id train_stage eval_kind
    pretrain_id="$(submit_stage pretrain)"
    if [[ "${algo}" == "ppo" ]]; then
        train_stage="train-ppo"
        eval_kind="evaluate-ppo"
    elif [[ "${algo}" == "dqn" ]]; then
        train_stage="train-dqn"
        eval_kind="evaluate-dqn"
    else
        echo "Unknown algo: ${algo}" >&2
        exit 1
    fi
    train_id="$(submit_stage "${train_stage}" "afterok:${pretrain_id}")"
    eval_id="$(submit_stage evaluate "afterok:${train_id}" "${eval_kind}")"
    baseline_id="$(submit_stage evaluate "" paper-baselines)"
    printf '%s,%s,%s,%s,%s,%s,%s\n' "${group}" "${algo}" "${dataset}" "${symbol}" "${RUN_NAME}" "${pretrain_id}" "${train_id}" "${eval_id}"
    printf '%s,%s,%s,%s,%s,%s,%s\n' "${group}" "${algo}" "${dataset}" "${symbol}" "${RUN_NAME}" "baseline" "${baseline_id}" >&2
}

REPO="$(repo_root)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
printf 'group,algo,dataset,symbol,run_name,pretrain_job,train_job,eval_job\n'

COMMON_PPO=(
    "TORCH_BATCH_SIZE=2048"
    "TORCH_LEARNING_RATE=0.0003"
    "PPO_EPOCHS=24"
    "PPO_ROLLOUTS_PER_EPOCH=128"
    "PPO_UPDATE_EPOCHS=6"
    "PPO_INITIAL_LOG_STD=-1.4"
    "PPO_INITIAL_SPREAD_BIAS=-0.70"
    "PPO_ENTROPY_COEF=0.006"
    "PPO_ENTROPY_COEF_FINAL=0.0002"
)
COMMON_DQN=(
    "TORCH_BATCH_SIZE=1024"
    "TORCH_LEARNING_RATE=0.00025"
    "TORCH_EPOCHS=10"
    "DQN_REPLAY_SIZE=250000"
    "DQN_MIN_REPLAY=4096"
    "DQN_UPDATE_INTERVAL=96"
    "DQN_TARGET_UPDATE_STEPS=1000"
    "DQN_EPSILON_START=0.50"
    "DQN_EPSILON_END=0.05"
    "DQN_EPSILON_DECAY=0.88"
)
SYNTH_000858=(
    "DATA_SOURCE=synthetic"
    "NUM_DAYS=16"
    "TRAIN_DAYS=10"
    "TEST_DAYS=6"
    "EVENTS_PER_DAY_OVERRIDE=60000"
    "EPISODE_LENGTH=2000"
    "MAX_TRAIN_EPISODES_PER_DAY=10"
    "MAX_EVAL_EPISODES_PER_DAY=10"
    "ORDER_FLOW_MEMORY=0.35"
    "VOLATILITY_CLUSTER_STRENGTH=0.45"
    "VOLATILITY_CLUSTER_PERSISTENCE=0.992"
)
REAL_COMMON=(
    "DATA_SOURCE=real"
    "REAL_DATA_ROOT=/cluster/work/math/piroth/mlfcs-gapa/data/processed"
    "REAL_EVENT_STRIDE=500"
    "REAL_START_TIME=09:30:00"
    "REAL_END_TIME=16:00:00"
    "NUM_DAYS=12"
    "TRAIN_DAYS=8"
    "TEST_DAYS=4"
    "EVENTS_PER_DAY_OVERRIDE=30000"
    "EPISODE_LENGTH=2000"
    "MAX_TRAIN_EPISODES_PER_DAY=8"
    "MAX_EVAL_EPISODES_PER_DAY=8"
)
REWARD_PNL_LOT1=(
    "TRADE_UNIT_OVERRIDE=1"
    "REWARD_MODE=hybrid"
    "REWARD_USE_DAMPENED_PNL=false"
    "REWARD_USE_TRADING_PNL=false"
    "REWARD_USE_INVENTORY_PENALTY=false"
    "REWARD_SPREAD_PENALTY_SCALE=0"
)
REWARD_INV_LOT1=(
    "TRADE_UNIT_OVERRIDE=1"
    "REWARD_MODE=hybrid"
    "REWARD_USE_DAMPENED_PNL=false"
    "REWARD_USE_TRADING_PNL=false"
    "REWARD_USE_INVENTORY_PENALTY=true"
    "REWARD_ZETA=0.000005"
    "REWARD_SPREAD_PENALTY_SCALE=0"
)
REWARD_TRDINV_LOT1=(
    "TRADE_UNIT_OVERRIDE=1"
    "REWARD_MODE=hybrid"
    "REWARD_USE_DAMPENED_PNL=false"
    "REWARD_USE_TRADING_PNL=true"
    "REWARD_TRADING_PNL_WEIGHT=0.25"
    "REWARD_USE_INVENTORY_PENALTY=true"
    "REWARD_ZETA=0.000005"
    "REWARD_SPREAD_PENALTY_SCALE=0"
)
BC_AS=(
    "BC_AS_INIT=true"
    "BC_AS_EPOCHS=4"
    "BC_AS_FREEZE_BACKBONE=true"
    "BC_AS_MAX_SAMPLES_PER_DAY=12000"
)

for reward_name in pnl_lot1 inv_lot1 trdinv_lot1; do
    reward_var="REWARD_${reward_name^^}"
    reward_var="${reward_var//_LOT1/_LOT1}"
    case "${reward_name}" in
        pnl_lot1) reward_env=("${REWARD_PNL_LOT1[@]}") ;;
        inv_lot1) reward_env=("${REWARD_INV_LOT1[@]}") ;;
        trdinv_lot1) reward_env=("${REWARD_TRDINV_LOT1[@]}") ;;
    esac
    submit_pipeline "rew_${reward_name}" ppo synth 000858 "${SYNTH_000858[@]}" "${COMMON_PPO[@]}" "${reward_env[@]}"
    submit_pipeline "rew_${reward_name}" dqn synth 000858 "${SYNTH_000858[@]}" "${COMMON_DQN[@]}" "${reward_env[@]}"
    submit_pipeline "rew_${reward_name}" ppo real AAPL "${REAL_COMMON[@]}" "${COMMON_PPO[@]}" "${reward_env[@]}"
    submit_pipeline "rew_${reward_name}" ppo real GOOGL "${REAL_COMMON[@]}" "${COMMON_PPO[@]}" "${reward_env[@]}"
done

submit_pipeline "rew_inv_lot1" dqn real AAPL "${REAL_COMMON[@]}" "${COMMON_DQN[@]}" "${REWARD_INV_LOT1[@]}"
submit_pipeline "rew_inv_lot1" dqn real GOOGL "${REAL_COMMON[@]}" "${COMMON_DQN[@]}" "${REWARD_INV_LOT1[@]}"

submit_pipeline "bc_as_inv_lot1" ppo synth 000858 "${SYNTH_000858[@]}" "${COMMON_PPO[@]}" "${REWARD_INV_LOT1[@]}" "${BC_AS[@]}"
submit_pipeline "bc_as_inv_lot1" dqn synth 000858 "${SYNTH_000858[@]}" "${COMMON_DQN[@]}" "${REWARD_INV_LOT1[@]}" "${BC_AS[@]}"
submit_pipeline "bc_as_inv_lot1" ppo real AAPL "${REAL_COMMON[@]}" "${COMMON_PPO[@]}" "${REWARD_INV_LOT1[@]}" "${BC_AS[@]}"
submit_pipeline "bc_as_inv_lot1" ppo real GOOGL "${REAL_COMMON[@]}" "${COMMON_PPO[@]}" "${REWARD_INV_LOT1[@]}" "${BC_AS[@]}"
submit_pipeline "bc_as_inv_lot1" dqn real AAPL "${REAL_COMMON[@]}" "${COMMON_DQN[@]}" "${REWARD_INV_LOT1[@]}" "${BC_AS[@]}"
