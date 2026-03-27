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
    local pretrain_run="$2"
    shift 2

    local train_id
    local evaluate_id
    local report_id

    train_id="$(
        submit_cmd "${run_name}" \
            DEPENDENCY="afterok:${PRETRAIN_JOB_ID}" \
            BACKBONE_RUN_NAME="${pretrain_run}" \
            "$@" \
            train
    )"
    echo "Submitted train for ${run_name}: ${train_id}"

    evaluate_id="$(
        submit_cmd "${run_name}" \
            DEPENDENCY="afterok:${train_id}" \
            BACKBONE_RUN_NAME="${pretrain_run}" \
            "$@" \
            evaluate
    )"
    echo "Submitted evaluate for ${run_name}: ${evaluate_id}"

    report_id="$(
        submit_cmd "${run_name}" \
            DEPENDENCY="afterok:${train_id}:${evaluate_id}" \
            BACKBONE_RUN_NAME="${pretrain_run}" \
            "$@" \
            report
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
    MAX_PRETRAIN_SAMPLES_PER_DAY=750000
    PRETRAIN_EPOCHS=12
    PRETRAIN_BATCH_SIZE=512
    PRETRAIN_NUM_WORKERS=8
    PRETRAIN_PREFETCH_FACTOR=4
    PRETRAIN_BALANCE_MODE=balanced_sampler_and_loss
    PRETRAIN_HORIZON=10
    TARGET_EPISODE_SECONDS=60
)

COMMON_RL=(
    TRAIN_TIME=1-00:00:00
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
    MAX_PRETRAIN_SAMPLES_PER_DAY=750000
    PRETRAIN_EPOCHS=12
    PRETRAIN_BATCH_SIZE=512
    PRETRAIN_NUM_WORKERS=8
    PRETRAIN_PREFETCH_FACTOR=4
    PRETRAIN_BALANCE_MODE=balanced_sampler_and_loss
    PRETRAIN_HORIZON=10
    TARGET_EPISODE_SECONDS=60
    REWARD_MODE=pnl_inventory
    GAMMA=0.99999
    GAE_LAMBDA=0.9995
    PPO_LR=2e-5
    PPO_MINIBATCH_SIZE=4096
    PPO_ROLLOUTS_PER_EPOCH=192
    PPO_EPOCHS=18
    PPO_UPDATES=3
    MAX_TRAIN_EPISODES_PER_DAY=192
    MAX_EVAL_EPISODES_PER_DAY=24
    NORMALIZE_ADVANTAGES=1
    GRADIENT_CLIP_NORM=0.5
    BACKBONE_TRAINABLE=1
    MAX_SPREAD_BPS=8.0
    MAX_BIAS_BPS=2.0
    MAX_INVENTORY=200
    ZETA=0.005
    ETA=0.0
)

PRETRAIN_RUN_NAME="${PRETRAIN_RUN_NAME:-euler_aapl_stage5_shared_pretrain}"

PRETRAIN_JOB_ID="$(
    submit_cmd "${PRETRAIN_RUN_NAME}" "${COMMON_PRETRAIN[@]}" pretrain
)"
echo "Submitted shared pretrain: ${PRETRAIN_JOB_ID}"

submit_variant "euler_aapl_stage5_ultra_plus" "${PRETRAIN_RUN_NAME}" "${COMMON_RL[@]}"
submit_variant "euler_aapl_stage5_risk_ramp" "${PRETRAIN_RUN_NAME}" "${COMMON_RL[@]}" REWARD_MODE=pnl_inventory_ramp ZETA_START=0.002 ZETA_END=0.02
submit_variant "euler_aapl_stage5_l1l2_inventory" "${PRETRAIN_RUN_NAME}" "${COMMON_RL[@]}" REWARD_MODE=pnl_inventory_l1l2 ZETA=0.003 ZETA_L2=0.003 ZETA_L1=0.0015
submit_variant "euler_aapl_stage5_competitive_quotes" "${PRETRAIN_RUN_NAME}" "${COMMON_RL[@]}" MAX_SPREAD_BPS=7.0 MAX_BIAS_BPS=3.0 MAX_INVENTORY=250 ZETA=0.004
submit_variant "euler_aapl_stage5_low_lr_long" "${PRETRAIN_RUN_NAME}" "${COMMON_RL[@]}" PPO_LR=1.5e-5 PPO_EPOCHS=22
