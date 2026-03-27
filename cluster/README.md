# Euler Submission Scripts

These scripts are designed for the ETH Euler cluster and default to the `ls_math` share.

## What they do

- `euler_pretrain.sbatch`: runs `lobmm.pretrain`
- `euler_train_ppo.sbatch`: runs PPO training and PPO test evaluation
- `euler_evaluate.sbatch`: runs the primitive non-RL baselines
- `euler_report.sbatch`: builds the report from an existing run directory
- `euler_suite.sbatch`: runs the end-to-end suite in one job
- `submit_euler.sh`: convenience wrapper for single jobs or a dependency-chained pipeline

The scripts do not require a `data/` symlink. By default they read from:

```bash
/cluster/work/math/$USER/mlfcs-gapa/data/processed
```

By default they write outputs to:

```bash
/cluster/project/math/$USER/mlfcs-gapa/artifacts
```

Override those with `DATA_DIR=...` or `OUTPUT_ROOT=...` if needed.

## Recommended usage

Submit the main pipeline with one GPU job per symbol and a final CPU report job:

```bash
RUN_NAME=euler_main SYMBOLS=AAPL,GOOGL cluster/submit_euler.sh pipeline
```

Submit the faster AAPL-only pipeline:

```bash
RUN_NAME=euler_main cluster/submit_euler.sh pipeline-aapl
```

Submit the representative medium-budget AAPL-only pipeline:

```bash
RUN_NAME=euler_mid cluster/submit_euler.sh pipeline-aapl-medium
```

Submit a single stage directly:

```bash
RUN_NAME=euler_pretrain SYMBOLS=AAPL sbatch cluster/euler_pretrain.sbatch
```

Run the suite with the faster defaults used here:

```bash
RUN_NAME=euler_suite cluster/submit_euler.sh suite
```

By default the suite keeps:

- `run_pretrain=1`
- `run_main_agents=1`
- `run_non_rl_baselines=1`
- `run_report=1`
- `run_rl_baselines=0`
- `run_ablations=0`
- `run_latency=0`

Turn those back on with environment variables such as `RUN_ABLATIONS=1`.

## Resource overrides

You can override Slurm resources from the wrapper without editing the scripts:

```bash
RUN_NAME=euler_main \
SYMBOLS=AAPL,GOOGL \
TRAIN_TIME=12:00:00 \
TRAIN_CPUS=10 \
TRAIN_MEM_PER_CPU=8G \
cluster/submit_euler.sh pipeline
```

Supported prefixes are `PRETRAIN_`, `TRAIN_`, `EVALUATE_`, `REPORT_`, and `SUITE_`.

For each prefix, the wrapper understands:

- `*_TIME`
- `*_CPUS`
- `*_MEM_PER_CPU`
- `*_GPUS`
- `*_TMP`
- `*_PARTITION`

The bundled sbatch defaults now use the full currently allowed Euler walltime for these partitions:

- `gpu.24h`: `2-00:00:00`
- `normal.24h`: `2-00:00:00`

Pretraining also checkpoints periodically and writes a resumable partial backbone so a timeout does not throw away all progress.

## Modes

- `MODE=full`: full dataset and uncapped episode budgets.
- `MODE=medium`: AAPL-friendly middle ground sized to finish a full pipeline in a few hours on Euler.
- `MODE=smoke`: tiny local plumbing check.

`pipeline-aapl-medium` is equivalent to `SYMBOLS=AAPL MODE=medium cluster/submit_euler.sh pipeline`.

## Python environment

By default, the jobs just use Euler's `python_cuda/3.11.6` module directly.
That avoids network-dependent package installation on compute nodes.

If you explicitly want an overlay virtualenv in scratch, set:

```bash
USE_OVERLAY_ENV=1
```
