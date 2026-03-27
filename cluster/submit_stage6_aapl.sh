#!/bin/bash

set -euo pipefail

repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.."
    pwd
}

submit_cmd() {
    local run_name="$1"
    shift
    local stage="${@: -1}"
    local -a env_vars=("${@:1:$#-1}")
    (
        cd "$(repo_root)"
        env RUN_NAME="${run_name}" SYMBOLS=AAPL MODE=full "${env_vars[@]}" cluster/submit_euler.sh "${stage}"
    )
}

submit_variant() {
    local run_name="$1"
    shift

    local train_id
    local evaluate_id
    local report_id

    train_id="$(
        submit_cmd "${run_name}" "$@" train
    )"
    echo "Submitted train for ${run_name}: ${train_id}"

    evaluate_id="$(
        submit_cmd "${run_name}" DEPENDENCY="afterok:${train_id}" "$@" evaluate
    )"
    echo "Submitted evaluate for ${run_name}: ${evaluate_id}"

    report_id="$(
        submit_cmd "${run_name}" DEPENDENCY="afterok:${train_id}:${evaluate_id}" "$@" report
    )"
    echo "Submitted report for ${run_name}: ${report_id}"
}

COMMON=(
    BACKBONE_RUN_NAME=euler_aapl_stage5_shared_pretrain
    TRAIN_TIME=18:00:00
    TRAIN_CPUS=8
    TRAIN_MEM_PER_CPU=6G
    TRAIN_GPUS=1
    TRAIN_PARTITION=gpu.24h
    TRAIN_DEVICE=cuda
    EVALUATE_TIME=02:00:00
    EVALUATE_CPUS=4
    EVALUATE_MEM_PER_CPU=4G
    EVALUATE_PARTITION=normal.4h
    EVALUATE_DEVICE=cpu
    REPORT_TIME=01:00:00
    REPORT_CPUS=2
    REPORT_MEM_PER_CPU=4G
    REPORT_PARTITION=normal.4h
    REPORT_DEVICE=cpu
    MAX_ROWS_PER_DAY=1000000
    TARGET_EPISODE_SECONDS=60
    REWARD_MODE=pnl_inventory
    GAMMA=0.99999
    GAE_LAMBDA=0.9995
    PPO_LR=3e-5
    PPO_MINIBATCH_SIZE=2048
    PPO_ROLLOUTS_PER_EPOCH=128
    PPO_EPOCHS=14
    PPO_UPDATES=2
    MAX_TRAIN_EPISODES_PER_DAY=128
    MAX_EVAL_EPISODES_PER_DAY=16
    NORMALIZE_ADVANTAGES=1
    GRADIENT_CLIP_NORM=0.5
    BACKBONE_TRAINABLE=1
    MAX_SPREAD_BPS=8.0
    MAX_BIAS_BPS=2.0
    MAX_INVENTORY=200
    ZETA=0.005
    ETA=0.0
)

submit_variant "euler_aapl_stage6_ultra_competitive" "${COMMON[@]}" MAX_SPREAD_BPS=7.0 MAX_BIAS_BPS=3.0 MAX_INVENTORY=250 ZETA=0.004
submit_variant "euler_aapl_stage6_ultra_competitive_ckpt" "${COMMON[@]}" MAX_SPREAD_BPS=7.0 MAX_BIAS_BPS=3.0 MAX_INVENTORY=250 ZETA=0.004 PPO_SELECT_BEST_MODEL=1 PPO_CHECKPOINT_EVERY=1 PPO_SELECTION_METRIC=pnl_mean
submit_variant "euler_aapl_stage6_ultra_control_ckpt" "${COMMON[@]}" PPO_SELECT_BEST_MODEL=1 PPO_CHECKPOINT_EVERY=1 PPO_SELECTION_METRIC=pnl_mean
