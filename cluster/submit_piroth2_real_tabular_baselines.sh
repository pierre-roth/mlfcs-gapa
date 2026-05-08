#!/bin/bash

set -euo pipefail

repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.."
    pwd
}

submit_kind() {
    local run_name="$1"
    local symbol="$2"
    local kind="$3"
    local dependency="${4:-}"
    local time_limit="${5:-04:00:00}"
    local -a dependency_args=()
    [[ -n "${dependency}" ]] && dependency_args+=(DEPENDENCY="${dependency}")
    local job_id
    job_id="$(
        env \
            ACCOUNT="${ACCOUNT:-ls_math}" \
            RUN_NAME="${run_name}" \
            SYMBOL="${symbol}" \
            MODE=full \
            KIND="${kind}" \
            EVALUATE_DEVICE=cpu \
            EVALUATE_TIME="${time_limit}" \
            EVALUATE_CPUS="${EVALUATE_CPUS:-8}" \
            EVALUATE_MEM_PER_CPU="${EVALUATE_MEM_PER_CPU:-8G}" \
            "${REAL_ENV[@]}" \
            "${TABLE2_COMMON[@]}" \
            "${dependency_args[@]}" \
            "${REPO}/cluster/submit_piroth2.sh" evaluate | tail -n 1
    )"
    printf '%s\n' "${job_id}"
}

submit_pair() {
    local symbol="$1"
    local run_name="${RUN_NAME_PREFIX:-piroth2}_table2_real_${symbol}_${STAMP}"
    local inventory_train inventory_eval lob_train lob_eval
    inventory_train="$(submit_kind "${run_name}" "${symbol}" train-inventory-rl "" "${TRAIN_TIME:-04:00:00}")"
    inventory_eval="$(submit_kind "${run_name}" "${symbol}" evaluate-inventory-rl "afterok:${inventory_train}" "${EVAL_TIME:-04:00:00}")"
    lob_train="$(submit_kind "${run_name}" "${symbol}" train-lob-rl "" "${TRAIN_TIME:-04:00:00}")"
    lob_eval="$(submit_kind "${run_name}" "${symbol}" evaluate-lob-rl "afterok:${lob_train}" "${EVAL_TIME:-04:00:00}")"
    printf 'tabular,%s,%s,%s,%s,%s,%s\n' "${symbol}" "${run_name}" "${inventory_train}" "${inventory_eval}" "${lob_train}" "${lob_eval}"
}

REPO="$(repo_root)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

TABLE2_COMMON=(
    "LOOKBACK=50"
    "PRETRAIN_HORIZON=10"
    "PRETRAIN_THRESHOLD=0.00001"
    "LOB_PRICE_Z_NORM=true"
    "STABLE_WINDOWS=10:00:00-14:30:00"
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
    "MAX_TRAIN_EPISODES_PER_DAY=12"
    "MAX_EVAL_EPISODES_PER_DAY=12"
    "TABULAR_EPOCHS=${TABULAR_EPOCHS:-80}"
    "TABULAR_ALPHA_START=${TABULAR_ALPHA_START:-0.25}"
    "TABULAR_ALPHA_END=${TABULAR_ALPHA_END:-0.03}"
    "TABULAR_EPSILON_START=${TABULAR_EPSILON_START:-0.35}"
    "TABULAR_EPSILON_END=${TABULAR_EPSILON_END:-0.03}"
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

SYMBOLS=(${SYMBOLS:-AAPL GOOGL})

printf 'kind,symbol,run_name,inventory_train,inventory_eval,lob_train,lob_eval\n'
for symbol in "${SYMBOLS[@]}"; do
    submit_pair "${symbol}"
done
