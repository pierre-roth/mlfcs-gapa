#!/bin/bash

set -euo pipefail

repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.."
    pwd
}

submit_pretrain() {
    local dataset="$1"
    local symbol="$2"
    local model_type="$3"
    shift 3

    local run_name="${RUN_NAME_PREFIX:-piroth2}_table1_${dataset}_${symbol}_${model_type}_${STAMP}"
    local -a env_args=(
        "ACCOUNT=${ACCOUNT:-ls_math}"
        "RUN_NAME=${run_name}"
        "SYMBOL=${symbol}"
        "MODE=full"
        "CREATE_PLOTS=false"
        "PRETRAIN_DEVICE=${PRETRAIN_DEVICE:-cuda}"
        "PRETRAIN_TIME=${PRETRAIN_TIME:-08:00:00}"
        "PRETRAIN_CPUS=${PRETRAIN_CPUS:-8}"
        "PRETRAIN_MEM_PER_CPU=${PRETRAIN_MEM_PER_CPU:-10G}"
        "PRETRAIN_MODEL_TYPE=${model_type}"
        "PRETRAIN_STABLE_WINDOWS_ONLY=true"
    )

    local job_id
    job_id="$(env "${env_args[@]}" "$@" "${REPO}/cluster/submit_piroth2.sh" pretrain | tail -n 1)"
    printf 'table1,%s,%s,%s,%s,%s\n' "${dataset}" "${symbol}" "${model_type}" "${run_name}" "${job_id}"
}

submit_table2_pipeline() {
    local dataset="$1"
    local symbol="$2"
    shift 2

    local run_name="${RUN_NAME_PREFIX:-piroth2}_table2_${dataset}_${symbol}_${STAMP}"
    local -a env_args=(
        "ACCOUNT=${ACCOUNT:-ls_math}"
        "RUN_NAME=${run_name}"
        "SYMBOL=${symbol}"
        "MODE=full"
        "CREATE_PLOTS=false"
        "PRETRAIN_DEVICE=${PRETRAIN_DEVICE:-cuda}"
        "TRAIN_PPO_DEVICE=${TRAIN_PPO_DEVICE:-cuda}"
        "TRAIN_DQN_DEVICE=${TRAIN_DQN_DEVICE:-cuda}"
        "EVALUATE_DEVICE=${EVALUATE_DEVICE:-cpu}"
        "PRETRAIN_TIME=${PRETRAIN_TIME:-08:00:00}"
        "TRAIN_PPO_TIME=${TRAIN_PPO_TIME:-08:00:00}"
        "TRAIN_DQN_TIME=${TRAIN_DQN_TIME:-08:00:00}"
        "EVALUATE_TIME=${EVALUATE_TIME:-04:00:00}"
        "REPORT_TIME=${REPORT_TIME:-01:00:00}"
        "PRETRAIN_MEM_PER_CPU=${PRETRAIN_MEM_PER_CPU:-10G}"
        "TRAIN_PPO_MEM_PER_CPU=${TRAIN_PPO_MEM_PER_CPU:-10G}"
        "TRAIN_DQN_MEM_PER_CPU=${TRAIN_DQN_MEM_PER_CPU:-10G}"
        "EVALUATE_MEM_PER_CPU=${EVALUATE_MEM_PER_CPU:-8G}"
    )

    local output
    output="$(env "${env_args[@]}" "$@" "${REPO}/cluster/submit_piroth2.sh" pipeline)"
    printf '%s\n' "${output}" >&2
    local pretrain ppo dqn ppo_eval dqn_eval baseline report
    pretrain="$(awk '/Submitted pretrain:/ {print $3}' <<<"${output}")"
    ppo="$(awk '/Submitted PPO train:/ {print $4}' <<<"${output}")"
    dqn="$(awk '/Submitted DQN train:/ {print $4}' <<<"${output}")"
    ppo_eval="$(awk '/Submitted PPO evaluation:/ {print $4}' <<<"${output}")"
    dqn_eval="$(awk '/Submitted DQN evaluation:/ {print $4}' <<<"${output}")"
    baseline="$(awk '/Submitted baseline evaluation:/ {print $4}' <<<"${output}")"
    report="$(awk '/Submitted report:/ {print $3}' <<<"${output}")"
    printf 'table2,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${dataset}" "${symbol}" "${run_name}" "${pretrain}" "${ppo}" "${dqn}" "${ppo_eval}" "${dqn_eval}" "${baseline}" "${report}"
}

REPO="$(repo_root)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

COMMON_PAPER=(
    "LOOKBACK=50"
    "PRETRAIN_HORIZON=10"
    "PRETRAIN_THRESHOLD=0.00001"
    "LOB_PRICE_Z_NORM=true"
    "STABLE_WINDOWS=10:00:00-14:30:00"
)

TABLE1_COMMON=(
    "${COMMON_PAPER[@]}"
    "TORCH_BATCH_SIZE=2048"
    "TORCH_LEARNING_RATE=0.0003"
    "TORCH_EPOCHS=8"
    "MAX_PRETRAIN_SAMPLES_PER_DAY=80000"
)

TABLE2_COMMON=(
    "${COMMON_PAPER[@]}"
    "EPISODE_LENGTH=2000"
    "REWARD_MODE=hybrid"
    "REWARD_ETA=0.5"
    "REWARD_ZETA=0.01"
    "REWARD_USE_DAMPENED_PNL=true"
    "REWARD_USE_TRADING_PNL=true"
    "REWARD_USE_INVENTORY_PENALTY=true"
    "REWARD_PNL_WEIGHT=1.0"
    "REWARD_TRADING_PNL_WEIGHT=1.0"
    "REWARD_INVENTORY_PENALTY_WEIGHT=1.0"
    "REWARD_SPREAD_PENALTY_WEIGHT=0.0"
    "MATCHING_MODE=multi_fill"
    "CONTINUOUS_ACTION_MODE=author"
    "TORCH_BATCH_SIZE=2048"
    "TORCH_LEARNING_RATE=0.0003"
    "TORCH_EPOCHS=10"
    "PPO_EPOCHS=10"
    "PPO_UPDATE_EPOCHS=4"
    "PPO_ROLLOUTS_PER_EPOCH=120"
    "PPO_SHUFFLE_EPISODES=true"
    "MAX_TRAIN_EPISODES_PER_DAY=12"
    "MAX_EVAL_EPISODES_PER_DAY=12"
    "DQN_REPLAY_SIZE=250000"
    "DQN_MIN_REPLAY=4096"
    "DQN_UPDATE_INTERVAL=64"
    "DQN_TARGET_UPDATE_STEPS=1000"
)

SYNTH_ENV=(
    "DATA_SOURCE=synthetic"
    "NUM_DAYS=21"
    "TRAIN_DAYS=10"
    "TEST_DAYS=11"
)

REAL_ENV=(
    "DATA_SOURCE=real"
    "REAL_DATA_ROOT=/cluster/work/math/piroth/mlfcs-gapa/data/processed"
    "REAL_EVENT_STRIDE=1"
    "REAL_START_TIME=10:00:00"
    "REAL_END_TIME=14:30:00"
    "NUM_DAYS=10"
    "TRAIN_DAYS=6"
    "TEST_DAYS=4"
    "EVENTS_PER_DAY_OVERRIDE=60000"
)

MODELS=(fclob convlob deeplob attnlob)
SYNTH_SYMBOLS=(000001 000858 002415)
REAL_SYMBOLS=(AAPL GOOGL)

printf 'table,dataset,symbol,model_or_run,run_name,job_ids...\n'
if [[ "${SKIP_SYNTHETIC:-false}" != "true" ]]; then
    for symbol in "${SYNTH_SYMBOLS[@]}"; do
        for model_type in "${MODELS[@]}"; do
            submit_pretrain synthetic "${symbol}" "${model_type}" "${TABLE1_COMMON[@]}" "${SYNTH_ENV[@]}"
        done
    done
fi
if [[ "${SKIP_REAL:-false}" != "true" ]]; then
    for symbol in "${REAL_SYMBOLS[@]}"; do
        for model_type in "${MODELS[@]}"; do
            submit_pretrain real "${symbol}" "${model_type}" "${TABLE1_COMMON[@]}" "${REAL_ENV[@]}"
        done
    done
fi

if [[ "${SKIP_SYNTHETIC:-false}" != "true" ]]; then
    for symbol in "${SYNTH_SYMBOLS[@]}"; do
        submit_table2_pipeline synthetic "${symbol}" "${TABLE2_COMMON[@]}" "${SYNTH_ENV[@]}"
    done
fi
if [[ "${SKIP_REAL:-false}" != "true" ]]; then
    for symbol in "${REAL_SYMBOLS[@]}"; do
        submit_table2_pipeline real "${symbol}" "${TABLE2_COMMON[@]}" "${REAL_ENV[@]}"
    done
fi
