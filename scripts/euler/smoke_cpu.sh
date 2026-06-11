#!/bin/bash
#SBATCH --job-name=mlfcs-smoke
#SBATCH --account=ls_math
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
PROJECT_CODE_DIR="${PROJECT_CODE_DIR:-${HOME}/projects/mlfcs-gapa}"
cd "${PROJECT_CODE_DIR}"
source .venv/bin/activate
source scripts/euler/wandb_env.sh

RUN_ROOT="${RUN_ROOT:-${SCRATCH:?}/mlfcs-gapa-smoke/${SLURM_JOB_ID}}"
DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}/data}"
export RUN_ROOT DATA_ROOT
mkdir -p "${RUN_ROOT}" "${DATA_ROOT}"

python -m pytest -q
mlfcs-gapa generate-synthetic \
  --output-dir "${DATA_ROOT}/synthetic-smoke" \
  --days 1 \
  --events-per-day 300 \
  --seed 101

python - <<'PY'
from pathlib import Path
import polars as pl

root = Path.home()  # deliberately unused; keep imports smoke-tested
path = Path(__import__("os").environ["DATA_ROOT"]) / "synthetic-smoke/000001/2019-11-01/orderbook.parquet"
df = pl.read_parquet(path)
assert df.shape == (300, 41), df.shape
print("synthetic smoke shape", df.shape)
PY

mlfcs-gapa run-synthetic-baselines \
  --output-dir "${RUN_ROOT}/synthetic-baseline-smoke" \
  --days 1 \
  --events-per-day 300 \
  --episode-events 200 \
  --tabular-episodes 3 \
  --seed 101 \
  "${WANDB_ARGS[@]}"

python - <<'PY'
from pathlib import Path
import os
import polars as pl

path = Path(os.environ["RUN_ROOT"]) / "synthetic-baseline-smoke/baseline_metrics.csv"
metrics = pl.read_csv(path)
assert metrics.height == 7, metrics
assert set(metrics["method"]) == {"Fixed_1", "Fixed_2", "Fixed_3", "Random", "AS", "Inv-RL", "LOB-RL"}
print("baseline smoke rows", metrics.height)
PY

mlfcs-gapa pretrain-synthetic-attn-lob \
  --output-dir "${RUN_ROOT}/synthetic-pretrain-smoke" \
  --events 220 \
  --epochs 1 \
  --batch-size 32 \
  --seed 101 \
  "${WANDB_ARGS[@]}"

python - <<'PY'
from pathlib import Path
import os
import polars as pl

path = Path(os.environ["RUN_ROOT"]) / "synthetic-pretrain-smoke/attn_lob_pretrain_metrics.csv"
metrics = pl.read_csv(path)
assert metrics.height == 1, metrics
assert metrics["model"][0] == "Attn-LOB"
model_path = Path(os.environ["RUN_ROOT"]) / "synthetic-pretrain-smoke/attn_lob_pretrain_model.pt"
assert model_path.exists(), model_path
print("pretrain smoke rows", metrics.height)
PY

mlfcs-gapa train-synthetic-ppo \
  --output-dir "${RUN_ROOT}/synthetic-ppo-smoke" \
  --events 240 \
  --episode-events 160 \
  --total-timesteps 32 \
  --n-steps 16 \
  --batch-size 8 \
  --n-epochs 1 \
  --seed 101 \
  "${WANDB_ARGS[@]}"

python - <<'PY'
from pathlib import Path
import os
import polars as pl

root = Path(os.environ["RUN_ROOT"]) / "synthetic-ppo-smoke"
metrics = pl.read_csv(root / "c_ppo_metrics.csv")
assert metrics.height == 1, metrics
assert metrics["method"][0] == "C-PPO"
assert (root / "c_ppo_model.zip").exists()
print("ppo smoke rows", metrics.height)
PY

mlfcs-gapa train-synthetic-ddqn \
  --output-dir "${RUN_ROOT}/synthetic-ddqn-smoke" \
  --events 240 \
  --episode-events 160 \
  --total-timesteps 32 \
  --learning-starts 8 \
  --buffer-size 128 \
  --batch-size 8 \
  --target-update-interval 16 \
  --seed 101 \
  "${WANDB_ARGS[@]}"

python - <<'PY'
from pathlib import Path
import os
import polars as pl

root = Path(os.environ["RUN_ROOT"]) / "synthetic-ddqn-smoke"
metrics = pl.read_csv(root / "d_dqn_metrics.csv")
losses = pl.read_csv(root / "d_dqn_losses.csv")
assert metrics.height == 1, metrics
assert metrics["method"][0] == "D-DQN"
assert losses.height > 0, losses
assert (root / "d_dqn_model.pt").exists()
print("ddqn smoke rows", metrics.height)
PY

mlfcs-gapa run-synthetic-latency-baselines \
  --output-dir "${RUN_ROOT}/synthetic-latency-smoke" \
  --latencies 1,5 \
  --days 1 \
  --events-per-day 300 \
  --episode-events 200 \
  --tabular-episodes 1 \
  --seed 101 \
  "${WANDB_ARGS[@]}"

mlfcs-gapa summarize-metrics \
  "${RUN_ROOT}/synthetic-latency-smoke/latency_metrics.csv" \
  --output-path "${RUN_ROOT}/synthetic-latency-smoke/summary_metrics.csv"

mlfcs-gapa plot-decision-trace \
  "${RUN_ROOT}/synthetic-latency-smoke/latency_trades.parquet" \
  --output-path "${RUN_ROOT}/synthetic-latency-smoke/decision_trace.png"

mlfcs-gapa plot-synthetic-attention \
  --output-path "${RUN_ROOT}/synthetic-latency-smoke/attention_heatmap.png" \
  --events 220 \
  --index 80 \
  --seed 101

python - <<'PY'
from pathlib import Path
import os
import polars as pl

root = Path(os.environ["RUN_ROOT"]) / "synthetic-latency-smoke"
metrics = pl.read_csv(root / "latency_metrics.csv")
summary = pl.read_csv(root / "summary_metrics.csv")
assert metrics.height == 10, metrics
assert summary.height == 5, summary
for name in ["latency_figure.png", "decision_trace.png", "attention_heatmap.png"]:
    path = root / name
    assert path.exists() and path.stat().st_size > 0, path
print("report smoke rows", metrics.height)
PY

mlfcs-gapa benchmark-runtime-synthetic \
  --output-path "${RUN_ROOT}/runtime-smoke/runtime_metrics.csv" \
  --events 220 \
  --episode-events 140 \
  --train-timesteps 16 \
  --device cpu \
  --seed 101 \
  "${WANDB_ARGS[@]}"

python - <<'PY'
from pathlib import Path
import os
import polars as pl

path = Path(os.environ["RUN_ROOT"]) / "runtime-smoke/runtime_metrics.csv"
metrics = pl.read_csv(path)
assert metrics.height == 7, metrics
assert set(metrics["method"]) == {"Random", "Fixed", "AS", "C-PPO", "D-DQN"}
assert set(metrics["phase"]) == {"infer", "train"}
print("runtime smoke rows", metrics.height)
PY
