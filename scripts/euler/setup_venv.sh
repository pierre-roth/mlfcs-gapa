#!/bin/bash
set -euo pipefail

module load stack/2024-06 python/3.12.8

PROJECT_CODE_DIR="${PROJECT_CODE_DIR:-${HOME}/projects/mlfcs-gapa}"
cd "${PROJECT_CODE_DIR}"
mkdir -p logs

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

python - <<'PY'
import sys
import torch
import polars
import gymnasium
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("polars", polars.__version__)
print("gymnasium", gymnasium.__version__)
print("cuda_available", torch.cuda.is_available())
PY
