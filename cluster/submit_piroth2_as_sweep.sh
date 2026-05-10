#!/bin/bash

set -euo pipefail

repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.."
    pwd
}

submit_as_sweep() {
    local dataset="$1"
    local symbol="$2"
    shift 2

    RUN_NAME="${RUN_NAME_PREFIX:-piroth2}_as_sweep_${dataset}_${symbol}_${STAMP}"
    local -a env_args=(
        "ACCOUNT=${ACCOUNT:-ls_math}"
        "RUN_NAME=${RUN_NAME}"
        "SYMBOL=${symbol}"
        "MODE=full"
        "CREATE_PLOTS=false"
        "EVALUATE_DEVICE=${EVALUATE_DEVICE:-cpu}"
        "EVALUATE_TIME=${EVALUATE_TIME:-02:00:00}"
        "EVALUATE_CPUS=${EVALUATE_CPUS:-8}"
        "EVALUATE_MEM_PER_CPU=${EVALUATE_MEM_PER_CPU:-16G}"
    )

    local job_id
    job_id="$(env "${env_args[@]}" "$@" KIND=as-sweep "${REPO}/cluster/submit_piroth2.sh" evaluate | tail -n 1)"
    printf '%s,%s,%s,%s\n' "${dataset}" "${symbol}" "${RUN_NAME}" "${job_id}"
}

REPO="$(repo_root)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

REAL_ENV=(
    "DATA_SOURCE=real"
    "REAL_DATA_ROOT=/cluster/work/math/piroth/mlfcs-gapa/data/processed"
    "REAL_EVENT_STRIDE=${REAL_EVENT_STRIDE:-1}"
    "REAL_START_TIME=${REAL_START_TIME:-10:00:00}"
    "REAL_END_TIME=${REAL_END_TIME:-16:00:00}"
    "NUM_DAYS=12"
    "TRAIN_DAYS=8"
    "TEST_DAYS=4"
    "EVENTS_PER_DAY_OVERRIDE=${EVENTS_PER_DAY_OVERRIDE:-60000}"
    "EPISODE_LENGTH=2000"
    "MAX_EVAL_EPISODES_PER_DAY=${MAX_EVAL_EPISODES_PER_DAY:-10}"
)

SYNTH_ENV=(
    "DATA_SOURCE=synthetic"
    "NUM_DAYS=16"
    "TRAIN_DAYS=10"
    "TEST_DAYS=6"
    "EVENTS_PER_DAY_OVERRIDE=60000"
    "EPISODE_LENGTH=2000"
    "MAX_EVAL_EPISODES_PER_DAY=${MAX_EVAL_EPISODES_PER_DAY:-10}"
    "ORDER_FLOW_MEMORY=0.35"
    "VOLATILITY_CLUSTER_STRENGTH=0.45"
    "VOLATILITY_CLUSTER_PERSISTENCE=0.992"
)

RUN_REAL="${RUN_REAL:-1}"
RUN_SYNTHETIC="${RUN_SYNTHETIC:-0}"
REAL_SYMBOLS=(${REAL_SYMBOLS:-AAPL GOOGL})
SYNTH_SYMBOLS=(${SYNTH_SYMBOLS:-000858})

printf 'dataset,symbol,run_name,job\n'
if [[ "${RUN_REAL}" == "1" ]]; then
    for symbol in "${REAL_SYMBOLS[@]}"; do
        submit_as_sweep real "${symbol}" "${REAL_ENV[@]}"
    done
fi
if [[ "${RUN_SYNTHETIC}" == "1" ]]; then
    for symbol in "${SYNTH_SYMBOLS[@]}"; do
        submit_as_sweep synthetic "${symbol}" "${SYNTH_ENV[@]}"
    done
fi
