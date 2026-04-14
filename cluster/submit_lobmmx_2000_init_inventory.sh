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
        env \
            RUN_NAME="${run_name}" \
            SYMBOLS=AAPL \
            MODE="${MODE:-medium}" \
            EULER_RUNNER_SCRIPT=cluster/euler_run_lobmmx.py \
            "${env_vars[@]}" \
            cluster/submit_euler.sh "${stage}"
    )
}

submit_variant() {
    local run_name="$1"
    shift

    local train_id
    local evaluate_id
    local report_id

    train_id="$(submit_cmd "${run_name}" "$@" train)"
    echo "Submitted train for ${run_name}: ${train_id}"

    evaluate_id="$(submit_cmd "${run_name}" "$@" DEPENDENCY="afterok:${train_id}" evaluate)"
    echo "Submitted evaluate for ${run_name}: ${evaluate_id}"

    report_id="$(submit_cmd "${run_name}" "$@" DEPENDENCY="afterok:${train_id}:${evaluate_id}" report)"
    echo "Submitted report for ${run_name}: ${report_id}"
}

COMMON_RL=(
    BACKBONE_RUN_NAME="${BACKBONE_RUN_NAME:-euler_lobmmx_aapl_stage2_shared_pretrain}"
    TRAIN_TIME="${TRAIN_TIME:-06:00:00}"
    TRAIN_CPUS="${TRAIN_CPUS:-8}"
    TRAIN_MEM_PER_CPU="${TRAIN_MEM_PER_CPU:-6G}"
    TRAIN_GPUS="${TRAIN_GPUS:-1}"
    TRAIN_PARTITION="${TRAIN_PARTITION:-gpu.24h}"
    TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
    EVALUATE_TIME="${EVALUATE_TIME:-01:00:00}"
    EVALUATE_CPUS="${EVALUATE_CPUS:-4}"
    EVALUATE_MEM_PER_CPU="${EVALUATE_MEM_PER_CPU:-4G}"
    EVALUATE_PARTITION="${EVALUATE_PARTITION:-normal.4h}"
    EVALUATE_DEVICE="${EVALUATE_DEVICE:-cpu}"
    REPORT_TIME="${REPORT_TIME:-00:30:00}"
    REPORT_CPUS="${REPORT_CPUS:-2}"
    REPORT_MEM_PER_CPU="${REPORT_MEM_PER_CPU:-4G}"
    REPORT_PARTITION="${REPORT_PARTITION:-normal.4h}"
    REPORT_DEVICE="${REPORT_DEVICE:-cpu}"
    MAX_ROWS_PER_DAY="${MAX_ROWS_PER_DAY:-200000}"
    TARGET_EPISODE_SECONDS=none
    EPISODE_LENGTH=2000
    REWARD_SCALE_MODE=spread
    RANDOM_INITIAL_INVENTORY=1
    INITIAL_INVENTORY_MAX=40
    ALLOW_TERMINAL_INVENTORY=1
    MAKER_REBATE_PER_SHARE=0.0013
    TAKER_FEE_PER_SHARE=0.0030
    FILL_MODEL=legacy
    GAMMA=0.999
    GAE_LAMBDA=0.995
    PPO_LR=3e-5
    PPO_MINIBATCH_SIZE=1024
    PPO_ROLLOUTS_PER_EPOCH=32
    PPO_EPOCHS=6
    PPO_UPDATES=2
    MAX_TRAIN_EPISODES_PER_DAY=64
    MAX_EVAL_EPISODES_PER_DAY=16
    NORMALIZE_ADVANTAGES=1
    GRADIENT_CLIP_NORM=0.5
    BACKBONE_TRAINABLE=1
    PPO_SELECT_BEST_MODEL=1
    PPO_CHECKPOINT_EVERY=1
    PPO_SELECTION_METRIC=pnl_mean
    QUOTE_SCALE_MODE=bps
    MAX_SPREAD_BPS=6.0
    MAX_BIAS_BPS=4.0
    MAX_INVENTORY_SKEW_BPS=3.0
    MAX_INVENTORY=125
)

submit_variant \
    "euler_lobmmx_stage4_2000_control" \
    "${COMMON_RL[@]}" \
    REWARD_MODE=trade_inventory \
    ZETA=0.0 \
    ETA=0.0 \
    TERMINAL_INVENTORY_REFERENCE=net_change \
    TERMINAL_INVENTORY_COST_SCALE=1.0

submit_variant \
    "euler_lobmmx_stage4_2000_excess_penalty" \
    "${COMMON_RL[@]}" \
    REWARD_MODE=trade_inventory_initial_excess \
    ZETA=0.01 \
    ETA=0.0 \
    REWARD_INVENTORY_POTENTIAL=0 \
    TERMINAL_INVENTORY_REFERENCE=excess_from_initial_abs \
    TERMINAL_INVENTORY_COST_SCALE=1.0

submit_variant \
    "euler_lobmmx_stage4_2000_excess_potential" \
    "${COMMON_RL[@]}" \
    REWARD_MODE=trade_inventory_initial_excess \
    ZETA=0.01 \
    ETA=0.25 \
    REWARD_INVENTORY_POTENTIAL=1 \
    TERMINAL_INVENTORY_REFERENCE=excess_from_initial_abs \
    TERMINAL_INVENTORY_COST_SCALE=1.0
