#!/bin/bash
#SBATCH --job-name=mlfcs-test-pipeline
#SBATCH --account=ls_math
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Small end-to-end test of the full replication pipeline. It runs the unit
# tests and then `run-full-synthetic-replication` at miniature scale through
# every stage the real job uses (Tables I-IV, Figures 2-4), and asserts the
# artifacts exist. If this passes, full_replication_gpu.sh should run clean.
#
# Usage (from $HOME/projects/mlfcs-gapa, after scripts/euler/setup_venv.sh):
#   sbatch scripts/euler/test_pipeline_cpu.sh

set -euo pipefail

module load stack/2024-06 python/3.12.8
PROJECT_CODE_DIR="${PROJECT_CODE_DIR:-${HOME}/projects/mlfcs-gapa}"
cd "${PROJECT_CODE_DIR}"
if [[ ! -d .venv ]]; then
  echo "missing .venv — run scripts/euler/setup_venv.sh first" >&2
  exit 1
fi
source .venv/bin/activate

RUN_ROOT="${RUN_ROOT:-${SCRATCH:-/tmp}/mlfcs-gapa-test-pipeline/${SLURM_JOB_ID:-manual}}"
export RUN_ROOT
SEED="${SEED:-7}"
mkdir -p "${RUN_ROOT}"
echo "run_root=${RUN_ROOT}"

python -m pytest -q

mlfcs-gapa run-full-synthetic-replication \
  --output-dir "${RUN_ROOT}/mini-replication" \
  --stocks 000001,000858 \
  --train-days 1 \
  --test-days 1 \
  --events-per-day 3000 \
  --episode-events 300 \
  --pretrain-epochs 1 \
  --pretrain-batch-size 64 \
  --agent-timesteps 256 \
  --tabular-episodes 3 \
  --latency-values 1,5 \
  --runtime-train-timesteps 16 \
  --ppo-n-envs 2 \
  --seed "${SEED}" \
  --device cpu

python - <<'PY'
import os
from pathlib import Path

import polars as pl

root = Path(os.environ["RUN_ROOT"]) / "mini-replication"

artifacts = [
    "README.md",
    "replication_config.md",
    "table_i_pretraining/table_i_pretrain_metrics.csv",
    "table_ii_overall/overall_metrics.csv",
    "table_ii_overall/overall_summary.csv",
    "figure_2_latency/latency_metrics.csv",
    "figure_2_latency/figure_2_latency.png",
    "figure_2_latency/figure_2_latency_paper.png",
    "table_iii_runtime/runtime_metrics.csv",
    "table_iv_ablation/ablation_metrics.csv",
    "table_iv_ablation/ablation_summary.csv",
    "figure_3_attention/figure_3_attention.png",
    "figure_4_decision_trace/figure_4_decision_trace.png",
]
for artifact in artifacts:
    path = root / artifact
    assert path.exists() and path.stat().st_size > 0, f"missing artifact: {path}"

table_i = pl.read_csv(root / "table_i_pretraining/table_i_pretrain_metrics.csv")
assert table_i.height == 4, table_i
assert table_i["param_matches_paper_report"].all(), table_i

overall = pl.read_csv(
    root / "table_ii_overall/overall_metrics.csv", schema_overrides={"stock": pl.Utf8}
)
methods = set(overall["method"].unique().to_list())
expected = {"C-PPO", "D-DQN", "Inv-RL", "LOB-RL", "AS", "Random", "Fixed_1", "Fixed_2", "Fixed_3"}
assert methods == expected, methods
assert set(overall["stock"].unique().to_list()) == {"000001", "000858"}

latency = pl.read_csv(
    root / "figure_2_latency/latency_metrics.csv", schema_overrides={"stock": pl.Utf8}
)
assert set(latency["latency_events"].unique().to_list()) == {1, 5}, latency

runtime = pl.read_csv(root / "table_iii_runtime/runtime_metrics.csv")
assert set(runtime["method"].unique().to_list()) == {"Random", "Fixed", "AS", "C-PPO", "D-DQN"}

ablation = pl.read_csv(
    root / "table_iv_ablation/ablation_metrics.csv", schema_overrides={"stock": pl.Utf8}
)
variants = set(ablation["variant"].unique().to_list())
assert variants == {"full", "without_lob", "without_attn_lob", "without_dynamic"}, variants
assert set(ablation["method"].unique().to_list()) == {"C-PPO", "D-DQN"}

print("pipeline test passed:", root)
PY
