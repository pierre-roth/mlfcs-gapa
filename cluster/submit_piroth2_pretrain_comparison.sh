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

    RUN_NAME="${RUN_NAME_PREFIX:-piroth2}_pretraincmp_${dataset}_${symbol}_${model_type}_${STAMP}"
    SYMBOL="${symbol}"

    local -a env_args=(
        "ACCOUNT=${ACCOUNT:-ls_math}"
        "RUN_NAME=${RUN_NAME}"
        "SYMBOL=${SYMBOL}"
        "MODE=full"
        "CREATE_PLOTS=false"
        "PRETRAIN_DEVICE=${PRETRAIN_DEVICE:-cuda}"
        "PRETRAIN_TIME=${PRETRAIN_TIME:-12:00:00}"
        "PRETRAIN_CPUS=${PRETRAIN_CPUS:-8}"
        "PRETRAIN_MEM_PER_CPU=${PRETRAIN_MEM_PER_CPU:-8G}"
        "PRETRAIN_MODEL_TYPE=${model_type}"
        "PRETRAIN_STABLE_WINDOWS_ONLY=true"
    )

    local job_id
    job_id="$(env "${env_args[@]}" "$@" "${REPO}/cluster/submit_piroth2.sh" pretrain | tail -n 1)"
    printf '%s,%s,%s,%s,%s\n' "${dataset}" "${symbol}" "${model_type}" "${RUN_NAME}" "${job_id}"
}

REPO="$(repo_root)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

COMMON_PRETRAIN=(
    "LOOKBACK=50"
    "PRETRAIN_HORIZON=10"
    "PRETRAIN_THRESHOLD=0.00001"
    "TORCH_BATCH_SIZE=${TORCH_BATCH_SIZE:-2048}"
    "TORCH_LEARNING_RATE=${TORCH_LEARNING_RATE:-0.0003}"
    "TORCH_EPOCHS=${TORCH_EPOCHS:-8}"
)

SYNTH_ENV=(
    "DATA_SOURCE=synthetic"
    "NUM_DAYS=16"
    "TRAIN_DAYS=10"
    "TEST_DAYS=6"
    "EVENTS_PER_DAY_OVERRIDE=60000"
    "MAX_PRETRAIN_SAMPLES_PER_DAY=80000"
    "ORDER_FLOW_MEMORY=0.35"
    "VOLATILITY_CLUSTER_STRENGTH=0.45"
    "VOLATILITY_CLUSTER_PERSISTENCE=0.992"
)

REAL_ENV=(
    "DATA_SOURCE=real"
    "REAL_DATA_ROOT=/cluster/work/math/piroth/mlfcs-gapa/data/processed"
    "REAL_EVENT_STRIDE=1"
    "REAL_START_TIME=09:30:00"
    "REAL_END_TIME=16:00:00"
    "NUM_DAYS=12"
    "TRAIN_DAYS=8"
    "TEST_DAYS=4"
)

MODELS=(fclob convlob deeplob attnlob)
SYNTH_SYMBOLS=(000001 000858 002415)
REAL_SYMBOLS=(AAPL GOOGL)
RUN_SYNTHETIC="${RUN_SYNTHETIC:-1}"
RUN_REAL="${RUN_REAL:-1}"

printf 'dataset,symbol,model_type,run_name,pretrain_job\n'
if [[ "${RUN_SYNTHETIC}" == "1" ]]; then
    for symbol in "${SYNTH_SYMBOLS[@]}"; do
        for model_type in "${MODELS[@]}"; do
            submit_pretrain synthetic "${symbol}" "${model_type}" "${COMMON_PRETRAIN[@]}" "${SYNTH_ENV[@]}"
        done
    done
fi
if [[ "${RUN_REAL}" == "1" ]]; then
    for symbol in "${REAL_SYMBOLS[@]}"; do
        for model_type in "${MODELS[@]}"; do
            submit_pretrain real "${symbol}" "${model_type}" "${COMMON_PRETRAIN[@]}" "${REAL_ENV[@]}"
        done
    done
fi
