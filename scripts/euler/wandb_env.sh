#!/bin/bash

WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_ENTITY="${WANDB_ENTITY:-piroth-ethz}"
WANDB_PROJECT="${WANDB_PROJECT:-mm-drl-lob}"

if [[ -z "${WANDB_GROUP:-}" ]]; then
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    WANDB_GROUP="${SLURM_ARRAY_JOB_ID}"
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    WANDB_GROUP="${SLURM_JOB_ID}"
  fi
fi

if [[ -z "${WANDB_RUN_NAME:-}" && -n "${SLURM_JOB_NAME:-}" ]]; then
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    WANDB_RUN_NAME="${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    WANDB_RUN_NAME="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
  fi
fi

WANDB_ARGS=()
case "${WANDB_ENABLED,,}" in
  1|true|yes|on)
    WANDB_ARGS+=(--wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}")
    if [[ -n "${WANDB_MODE:-}" ]]; then
      WANDB_ARGS+=(--wandb-mode "${WANDB_MODE}")
    fi
    if [[ -n "${WANDB_GROUP:-}" ]]; then
      WANDB_ARGS+=(--wandb-group "${WANDB_GROUP}")
    fi
    if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
      WANDB_ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
    fi
    ;;
esac
