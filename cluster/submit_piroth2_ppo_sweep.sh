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

    local -a env_args=(
        "ACCOUNT=${ACCOUNT:-ls_math}"
        "KIND=${KIND:-}"
        "RUN_NAME=${RUN_NAME}"
        "SYMBOL=${SYMBOL}"
        "SEED=${SEED}"
        "MODE=${MODE}"
        "EVENTS_PER_DAY_OVERRIDE=${EVENTS_PER_DAY_OVERRIDE}"
        "EPISODE_LENGTH=${EPISODE_LENGTH}"
        "MAX_EVAL_EPISODES_PER_DAY=${MAX_EVAL_EPISODES_PER_DAY}"
        "TORCH_BATCH_SIZE=${TORCH_BATCH_SIZE}"
        "CREATE_PLOTS=${CREATE_PLOTS}"
        "PRETRAIN_DEVICE=${PRETRAIN_DEVICE}"
        "TRAIN_PPO_DEVICE=${TRAIN_PPO_DEVICE}"
        "EVALUATE_DEVICE=${EVALUATE_DEVICE}"
        "PRETRAIN_TIME=${PRETRAIN_TIME}"
        "TRAIN_PPO_TIME=${TRAIN_PPO_TIME}"
        "EVALUATE_TIME=${EVALUATE_TIME}"
        "PRETRAIN_CPUS=${PRETRAIN_CPUS}"
        "TRAIN_PPO_CPUS=${TRAIN_PPO_CPUS}"
        "EVALUATE_CPUS=${EVALUATE_CPUS}"
        "PRETRAIN_MEM_PER_CPU=${PRETRAIN_MEM_PER_CPU}"
        "TRAIN_PPO_MEM_PER_CPU=${TRAIN_PPO_MEM_PER_CPU}"
        "EVALUATE_MEM_PER_CPU=${EVALUATE_MEM_PER_CPU}"
    )

    if [[ -n "${dependency}" ]]; then
        env_args+=("DEPENDENCY=${dependency}")
    fi
    env "${env_args[@]}" "${REPO}/cluster/submit_piroth2.sh" "${stage}"
}

REPO="$(repo_root)"
export REPO

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SYMBOLS="${SYMBOLS:-000001 000858 002415}"
SEEDS="${SEEDS:-11 17}"
MODE="${MODE:-medium}"
EVENTS_PER_DAY_OVERRIDE="${EVENTS_PER_DAY_OVERRIDE:-30000}"
EPISODE_LENGTH="${EPISODE_LENGTH:-2000}"
MAX_EVAL_EPISODES_PER_DAY="${MAX_EVAL_EPISODES_PER_DAY:-3}"
TORCH_BATCH_SIZE="${TORCH_BATCH_SIZE:-1024}"
CREATE_PLOTS="${CREATE_PLOTS:-false}"
PRETRAIN_DEVICE="${PRETRAIN_DEVICE:-cuda}"
TRAIN_PPO_DEVICE="${TRAIN_PPO_DEVICE:-cuda}"
EVALUATE_DEVICE="${EVALUATE_DEVICE:-cpu}"
PRETRAIN_TIME="${PRETRAIN_TIME:-08:00:00}"
TRAIN_PPO_TIME="${TRAIN_PPO_TIME:-12:00:00}"
EVALUATE_TIME="${EVALUATE_TIME:-02:00:00}"
PRETRAIN_CPUS="${PRETRAIN_CPUS:-8}"
TRAIN_PPO_CPUS="${TRAIN_PPO_CPUS:-8}"
EVALUATE_CPUS="${EVALUATE_CPUS:-4}"
PRETRAIN_MEM_PER_CPU="${PRETRAIN_MEM_PER_CPU:-8G}"
TRAIN_PPO_MEM_PER_CPU="${TRAIN_PPO_MEM_PER_CPU:-8G}"
EVALUATE_MEM_PER_CPU="${EVALUATE_MEM_PER_CPU:-8G}"

printf 'Submitting PPO seed sweep: symbols=(%s), seeds=(%s), stamp=%s\n' "${SYMBOLS}" "${SEEDS}" "${STAMP}"
printf 'run_name,pretrain_job,ppo_job,eval_job\n'

for symbol_value in ${SYMBOLS}; do
    for seed_value in ${SEEDS}; do
        export SYMBOL="${symbol_value}"
        export SEED="${seed_value}"
        export RUN_NAME="${RUN_NAME_PREFIX:-piroth2_ppo_seed}${SEED}_${SYMBOL}_${STAMP}"

        pretrain_id="$(submit_stage pretrain)"
        ppo_id="$(submit_stage train-ppo "afterok:${pretrain_id}")"
        eval_id="$(KIND=evaluate-ppo submit_stage evaluate "afterok:${ppo_id}")"
        printf '%s,%s,%s,%s\n' "${RUN_NAME}" "${pretrain_id}" "${ppo_id}" "${eval_id}"
    done
done
