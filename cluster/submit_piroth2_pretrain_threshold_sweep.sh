#!/bin/bash

set -euo pipefail

repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.."
    pwd
}

submit_pretrain() {
    local symbol="$1"
    local model_type="$2"
    local threshold_label="$3"
    local threshold="$4"
    local weight_mode="$5"

    RUN_NAME="${RUN_NAME_PREFIX:-piroth2}_pretrainthr_real_${symbol}_${model_type}_${threshold_label}_${weight_mode}_${STAMP}"
    SYMBOL="${symbol}"
    local lookback
    case "${model_type}" in
        fclob) lookback=100 ;;
        convlob) lookback=1024 ;;
        deeplob) lookback=100 ;;
        attnlob) lookback=50 ;;
        *) echo "Unknown pretrain model: ${model_type}" >&2; return 1 ;;
    esac

    local -a env_args=(
        "ACCOUNT=${ACCOUNT:-ls_math}"
        "RUN_NAME=${RUN_NAME}"
        "SYMBOL=${SYMBOL}"
        "MODE=full"
        "CREATE_PLOTS=false"
        "PRETRAIN_DEVICE=${PRETRAIN_DEVICE:-cuda}"
        "PRETRAIN_TIME=${PRETRAIN_TIME:-1-00:00:00}"
        "PRETRAIN_CPUS=${PRETRAIN_CPUS:-8}"
        "PRETRAIN_MEM_PER_CPU=${PRETRAIN_MEM_PER_CPU:-16G}"
        "PRETRAIN_MODEL_TYPE=${model_type}"
        "PRETRAIN_STABLE_WINDOWS_ONLY=true"
        "DATA_SOURCE=real"
        "REAL_DATA_ROOT=/cluster/work/math/piroth/mlfcs-gapa/data/processed"
        "REAL_EVENT_STRIDE=${REAL_EVENT_STRIDE:-1}"
        "REAL_BUILD_DEPTH_CUBE=${REAL_BUILD_DEPTH_CUBE:-false}"
        "REAL_START_TIME=${REAL_START_TIME:-09:30:00}"
        "REAL_END_TIME=${REAL_END_TIME:-16:00:00}"
        "NUM_DAYS=${NUM_DAYS:-12}"
        "TRAIN_DAYS=${TRAIN_DAYS:-8}"
        "TEST_DAYS=${TEST_DAYS:-4}"
        "LOOKBACK=${LOOKBACK:-${lookback}}"
        "PRETRAIN_HORIZON=${PRETRAIN_HORIZON:-10}"
        "PRETRAIN_THRESHOLD=${threshold}"
        "PRETRAIN_CLASS_WEIGHT_MODE=${weight_mode}"
        "TORCH_BATCH_SIZE=${TORCH_BATCH_SIZE:-8192}"
        "TORCH_LEARNING_RATE=${TORCH_LEARNING_RATE:-0.0003}"
        "TORCH_EPOCHS=${TORCH_EPOCHS:-6}"
    )

    local job_id
    job_id="$(env "${env_args[@]}" "${REPO}/cluster/submit_piroth2.sh" pretrain | tail -n 1)"
    printf '%s,%s,%s,%s,%s,%s\n' "${symbol}" "${model_type}" "${threshold}" "${weight_mode}" "${RUN_NAME}" "${job_id}"
}

REPO="$(repo_root)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

SYMBOLS=(${SYMBOLS:-AAPL GOOGL})
MODELS=(${MODELS:-fclob convlob deeplob attnlob})
WEIGHT_MODES=(${WEIGHT_MODES:-none balanced})

THRESHOLD_LABELS=(${THRESHOLD_LABELS:-t0 t2p5e6 t5e6 t1e5 t2e5 t5e5})
THRESHOLD_VALUES=(${THRESHOLD_VALUES:-0 0.0000025 0.000005 0.00001 0.00002 0.00005})

if [[ "${#THRESHOLD_LABELS[@]}" -ne "${#THRESHOLD_VALUES[@]}" ]]; then
    echo "THRESHOLD_LABELS and THRESHOLD_VALUES must have the same length" >&2
    exit 1
fi

printf 'symbol,model_type,threshold,weight_mode,run_name,pretrain_job\n'
for symbol in "${SYMBOLS[@]}"; do
    for model_type in "${MODELS[@]}"; do
        for idx in "${!THRESHOLD_VALUES[@]}"; do
            for weight_mode in "${WEIGHT_MODES[@]}"; do
                submit_pretrain "${symbol}" "${model_type}" "${THRESHOLD_LABELS[$idx]}" "${THRESHOLD_VALUES[$idx]}" "${weight_mode}"
            done
        done
    done
done
