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
        env RUN_NAME="${run_name}" SYMBOLS=AAPL MODE=full EULER_RUNNER_SCRIPT=cluster/euler_run_lobmmx.py "${env_vars[@]}" cluster/submit_euler.sh "${stage}"
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

COMMON_PRETRAIN=(
    PRETRAIN_TIME=08:00:00
    PRETRAIN_CPUS=8
    PRETRAIN_MEM_PER_CPU=6G
    PRETRAIN_GPUS=1
    PRETRAIN_PARTITION=gpu.24h
    PRETRAIN_DEVICE=cuda
    MAX_ROWS_PER_DAY=1000000
    MAX_PRETRAIN_SAMPLES_PER_DAY=500000
    PRETRAIN_EPOCHS=8
    PRETRAIN_BATCH_SIZE=512
    PRETRAIN_NUM_WORKERS=8
    PRETRAIN_PREFETCH_FACTOR=4
    PRETRAIN_BALANCE_MODE=balanced_sampler_and_loss
    PRETRAIN_TASK_MODE=mm_multitask
    PRETRAIN_HORIZON=10
    PRETRAIN_SPREAD_ALPHA_TICKS=0.25
    PRETRAIN_FLOW_ALPHA=100
    TARGET_EPISODE_SECONDS=60
)

COMMON_RL=(
    BACKBONE_RUN_NAME=euler_lobmmx_aapl_shared_pretrain
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
    REWARD_MODE=trade_inventory
    REWARD_SCALE_MODE=spread
    RANDOM_INITIAL_INVENTORY=1
    INITIAL_INVENTORY_MAX=125
    ALLOW_TERMINAL_INVENTORY=1
    MAKER_REBATE_PER_SHARE=0.0013
    TAKER_FEE_PER_SHARE=0.0030
    GAMMA=0.99999
    GAE_LAMBDA=0.9995
    PPO_LR=3e-5
    PPO_MINIBATCH_SIZE=2048
    PPO_ROLLOUTS_PER_EPOCH=128
    PPO_EPOCHS=14
    PPO_UPDATES=2
    MAX_TRAIN_EPISODES_PER_DAY=128
    MAX_EVAL_EPISODES_PER_DAY=32
    NORMALIZE_ADVANTAGES=1
    GRADIENT_CLIP_NORM=0.5
    BACKBONE_TRAINABLE=1
    PPO_SELECT_BEST_MODEL=1
    PPO_CHECKPOINT_EVERY=1
    PPO_SELECTION_METRIC=reward_mean
    QUOTE_SCALE_MODE=bps
    MAX_SPREAD_BPS=7.0
    MAX_BIAS_BPS=3.0
    MAX_INVENTORY_SKEW_BPS=4.0
    MAX_INVENTORY=250
    ZETA=0.004
    ETA=0.0
)

PRETRAIN_JOB_ID="$(
    submit_cmd euler_lobmmx_aapl_shared_pretrain "${COMMON_PRETRAIN[@]}" pretrain
)"
echo "Submitted creative shared pretrain: ${PRETRAIN_JOB_ID}"

submit_variant "euler_lobmmx_aapl_spread_base" DEPENDENCY="afterok:${PRETRAIN_JOB_ID}" "${COMMON_RL[@]}"
submit_variant "euler_lobmmx_aapl_ticks_base" DEPENDENCY="afterok:${PRETRAIN_JOB_ID}" "${COMMON_RL[@]}" REWARD_SCALE_MODE=ticks
submit_variant "euler_lobmmx_aapl_spread_alpha" DEPENDENCY="afterok:${PRETRAIN_JOB_ID}" "${COMMON_RL[@]}" MAX_BIAS_BPS=4.0 MAX_INVENTORY_SKEW_BPS=3.0
