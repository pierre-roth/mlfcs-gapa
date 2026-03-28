#!/bin/bash

set -euo pipefail

repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.."
    pwd
}

submit_pipeline() {
    local seed="$1"
    local run_name="euler_full_stage8_competitive_seed${seed}"
    (
        cd "$(repo_root)"
        env \
            RUN_NAME="${run_name}" \
            SYMBOLS=AAPL,GOOGL \
            MODE=full \
            SEED="${seed}" \
            PRETRAIN_TIME=08:00:00 \
            TRAIN_TIME=18:00:00 \
            EVALUATE_TIME=02:00:00 \
            REPORT_TIME=01:00:00 \
            PRETRAIN_CPUS=8 \
            PRETRAIN_MEM_PER_CPU=6G \
            PRETRAIN_GPUS=1 \
            PRETRAIN_PARTITION=gpu.24h \
            TRAIN_CPUS=8 \
            TRAIN_MEM_PER_CPU=6G \
            TRAIN_GPUS=1 \
            TRAIN_PARTITION=gpu.24h \
            EVALUATE_CPUS=4 \
            EVALUATE_MEM_PER_CPU=4G \
            EVALUATE_PARTITION=normal.4h \
            REPORT_CPUS=2 \
            REPORT_MEM_PER_CPU=4G \
            REPORT_PARTITION=normal.4h \
            MAX_ROWS_PER_DAY=1000000 \
            MAX_PRETRAIN_SAMPLES_PER_DAY=750000 \
            PRETRAIN_EPOCHS=12 \
            PRETRAIN_BATCH_SIZE=512 \
            PRETRAIN_NUM_WORKERS=8 \
            PRETRAIN_PREFETCH_FACTOR=4 \
            PRETRAIN_BALANCE_MODE=balanced_sampler_and_loss \
            PRETRAIN_HORIZON=10 \
            TARGET_EPISODE_SECONDS=60 \
            REWARD_MODE=pnl_inventory \
            GAMMA=0.99999 \
            GAE_LAMBDA=0.9995 \
            PPO_LR=3e-5 \
            PPO_MINIBATCH_SIZE=2048 \
            PPO_ROLLOUTS_PER_EPOCH=128 \
            PPO_EPOCHS=14 \
            PPO_UPDATES=2 \
            MAX_TRAIN_EPISODES_PER_DAY=128 \
            MAX_EVAL_EPISODES_PER_DAY=16 \
            NORMALIZE_ADVANTAGES=1 \
            GRADIENT_CLIP_NORM=0.5 \
            BACKBONE_TRAINABLE=1 \
            MAX_SPREAD_BPS=7.0 \
            MAX_BIAS_BPS=3.0 \
            MAX_INVENTORY=250 \
            ZETA=0.004 \
            ETA=0.0 \
            cluster/submit_euler.sh pipeline
    )
}

submit_pipeline 29
submit_pipeline 41
