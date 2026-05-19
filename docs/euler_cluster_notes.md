# ETH Euler Cluster Notes for This Project

This note summarizes the current official ETH Euler documentation and live checks from `ssh euler` for running the market-making replication.

Current date of research: 2026-05-18.

## Primary Sources

- Getting started: https://docs.hpc.ethz.ch/tutorials/getting-started/
- SSH: https://docs.hpc.ethz.ch/connections/ssh/
- Login nodes: https://docs.hpc.ethz.ch/hardware/login_nodes/
- Slurm: https://docs.hpc.ethz.ch/batchsystem/slurm/
- First job tutorial: https://docs.hpc.ethz.ch/tutorials/first-job/
- Storage: https://docs.hpc.ethz.ch/hardware/storage/
- Software stack: https://docs.hpc.ethz.ch/software/software-stack/
- Environment modules: https://docs.hpc.ethz.ch/software/environment-modules/
- Python: https://docs.hpc.ethz.ch/software/proglang/python/
- Conda: https://docs.hpc.ethz.ch/software/package-managers/conda/
- PyTorch: https://docs.hpc.ethz.ch/software/machine_learning/pytorch/
- GPU nodes: https://docs.hpc.ethz.ch/hardware/gpu_nodes/
- Apptainer: https://docs.hpc.ethz.ch/software/apptainer/
- Network/VPN: https://docs.hpc.ethz.ch/hardware/network/

## Live Account Check

Command used:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 euler '...'
```

Observed:

- Login works with the local SSH alias `euler`.
- Host reached: `eu-login-05`.
- SSH login user: `piroth`.
- Home: `/cluster/home/piroth`.
- Scratch: `/cluster/scratch/piroth`.
- The account/share requested by the user, `ls_math`, is not the SSH login username. It is a Slurm shareholder account/share.
- `my_share_info` reports:
  - member of `ls_krausea`
  - member of `ls_math`
  - system default Slurm share is `ls_math`
  - `ls_math` contains 1504 CPU cores, 5248 GiB system RAM, and 108 GPUs
  - `ls_krausea` contains 480 CPU cores, 1568 GiB system RAM, and 44 GPUs
- Since `ls_math` is already the system default share, jobs should still specify `#SBATCH --account=ls_math` for clarity and reproducibility.

Useful commands:

```bash
ssh euler
my_share_info
lquota
squeue -u "$USER"
myjobs
sacct --format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,MaxRSS
```

## Access Rules

- Euler is accessible via SSH at `euler.ethz.ch`.
- ETH docs say Euler is only accessible from the ETH network or via ETH VPN.
- If connecting from outside ETH and SSH fails, connect to ETH VPN first.
- First login may require accepting cluster usage rules.
- Euler accounts are personal. Do not use someone else's ETH account.

Recommended SSH command:

```bash
ssh piroth@euler.ethz.ch
```

The local alias also works here:

```bash
ssh euler
```

## Login Nodes

- Login nodes are for:
  - file management
  - editing
  - compiling small code
  - submitting jobs
  - checking job status
- Do not run long or resource-intensive work on login nodes.
- Use Slurm for compute.
- Euler has 50 login nodes behind the `euler.ethz.ch` load balancer.
- Reconnecting may land on a different login node. For persistent work, prefer Slurm interactive jobs, not detached login-node sessions.

## Slurm

Euler uses Slurm. Official docs state Slurm 25.05 since 2026-01-13; live check returned `slurm 25.05.5`.

Main commands:

```bash
sbatch job.sh
srun --pty bash
squeue -u "$USER"
myjobs
scancel <jobid>
sacct
my_share_info
get_inefficient_jobs
```

ETH guidance:

- Request resources accurately.
- Smaller resource requests usually start sooner.
- Do not specify partitions unless strictly necessary.
- Do not specify CPU/GPU types unless strictly necessary.
- GPU nodes are available only to shareholder groups that invested in GPUs.

Common `sbatch` options:

```bash
#SBATCH --job-name=<name>
#SBATCH --account=ls_math
#SBATCH --time=HH:MM:SS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<n>
#SBATCH --mem-per-cpu=<size>
#SBATCH --gpus-per-task=<n>
#SBATCH --tmp=<size>
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
```

Interactive CPU session:

```bash
srun --account=ls_math --time=00:30:00 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=4G --pty bash
```

Interactive GPU session:

```bash
srun --account=ls_math --time=01:00:00 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=4G --gpus-per-task=1 --pty bash
```

Use GPU only when needed. Data preprocessing and simulator debugging should start on CPU.

## Storage

Official storage layout:

| Type | Path | Env var | Use |
|---|---|---|---|
| Home | `/cluster/home/<username>` | `$HOME` | private long-term important files |
| Scratch | `/cluster/scratch/<username>` | `$SCRATCH` | temporary large data |
| Project | `/cluster/project/<group>` | none | long-term shared group storage |
| Work | `/cluster/work/<group>` | none | high-performance shared group storage |
| Tmp | `/tmp` on compute node | `$TMPDIR` | node-local job scratch |

Live quota for this account:

- `/cluster/home/piroth`
  - used: 24.05 GB
  - soft quota: 45.00 GB
  - hard quota: 50.00 GB
  - files used: 18,065
  - file hard quota: 500,000
- `/cluster/scratch/piroth`
  - used: 18.97 GB
  - soft quota: 2.50 TB
  - hard quota: 2.70 TB
  - files used: 53,341
  - file hard quota: 1,500,000

Group storage live check:

- `/cluster/project/ls_math`: missing
- `/cluster/work/ls_math`: missing
- `/cluster/project/ls_krausea`: missing
- `/cluster/work/ls_krausea`: missing

Implication for this project:

- Keep source code and small configs in `$HOME`.
- Put datasets, generated LOB data, training artifacts, checkpoints, and logs in `$SCRATCH`.
- For large file fanout, avoid millions of small files. Prefer Parquet, Arrow IPC, HDF5, NPZ, or tar shards.
- Scratch is automatically purged. Official docs say files older than about 15 days are deleted and scratch is not backed up.
- If a Slurm job needs fast temporary local I/O, request node-local tmp:

```bash
#SBATCH --tmp=100G
```

Then use `$TMPDIR` inside the job and copy results back before exit.

## Software Environment

Modules:

- Euler uses Lua/Lmod environment modules.
- No stack is loaded by default.
- Load a stack before loading packages.

Useful module commands:

```bash
module list
module avail
module spider <name>
module load stack/2024-06
module purge
module reset
```

Python:

- Official docs list Python 3.12.8 as:

```bash
module load stack/2024-06 python/3.12.8
```

- Live check confirms:

```text
Python 3.12.8
/cluster/software/stacks/2024-06/spack/opt/spack/.../python-3.12.8.../bin/python
```

`uv`:

- ETH Python docs mention `uv` as a package/project manager.
- Live check found no `uv` command and no `uv` module.
- If we want to use `uv` on Euler, install it under `$HOME` or use the standalone installer. Otherwise use `python -m venv`.

Conda:

- ETH docs say there is no centrally installed Conda.
- They warn Conda creates many small files and is a poor fit for Lustre.
- If Conda is required, use Home for small envs, Scratch for temporary envs, or Apptainer to package the env into a container.
- For this project, prefer either:
  - Python module plus venv, or
  - install `uv` once and keep `.venv` under `$SCRATCH`.

Recommended first setup on Euler:

```bash
ssh euler
module load stack/2024-06 python/3.12.8
mkdir -p "$SCRATCH/mlfcs-gapa"
cd "$SCRATCH/mlfcs-gapa"
git clone <repo-url> .
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If using `uv` after installing it:

```bash
module load stack/2024-06 python/3.12.8
mkdir -p "$SCRATCH/mlfcs-gapa"
cd "$SCRATCH/mlfcs-gapa"
uv sync
```

## PyTorch and GPUs

ETH PyTorch docs recommend installing PyTorch from the official PyTorch instructions:

- CUDA version for NVIDIA GPUs.
- ROCm version for AMD GPUs.
- CPU version if no GPU is needed.

Live module check:

- `module avail cuda` shows `cuda/13.0.2`.
- No `pytorch` module was found.
- No `uv` module was found.

GPU access:

- The user is in group `ID-HPC-EULERGPU`.
- The `ls_math` share has 108 GPUs according to `my_share_info`.
- Euler GPU models listed in docs include RTX 2080 Ti, TITAN RTX, Quadro RTX 6000, RTX 3090, Tesla A100, RTX 4090, RTX PRO 6000, and AMD MI300A.
- Do not request a specific GPU model unless the job requires it.

For our project:

- Data preprocessing and baseline simulation can run CPU-only.
- Attn-LOB pretraining and PPO training should use one GPU when available.
- Start with one-GPU jobs. Multi-GPU is not needed initially.

## Apptainer

Live check:

- `/usr/bin/apptainer` exists.
- The account did not show `ID-HPC-SINGULARITY` in `id`, so container execution may require requesting access.
- Official docs say run:

```bash
get-access
```

Then choose:

```text
[2] Apptainer (Singularity)
```

If using Apptainer:

```bash
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer"
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}"
apptainer exec --nv --bind "$SCRATCH:$SCRATCH" image.sif python train.py
```

For this project, Apptainer is optional. It becomes useful if Python package installation creates too many files or CUDA/PyTorch compatibility becomes painful.

## Recommended Repository Layout on Euler

Use:

```text
$HOME/src/mlfcs-gapa              small clone or bare config only
$SCRATCH/mlfcs-gapa               active working clone, venv, data, outputs
$SCRATCH/mlfcs-gapa/data          synthetic/public LOB data
$SCRATCH/mlfcs-gapa/runs          checkpoints, tensorboard, metrics
$SCRATCH/mlfcs-gapa/logs          Slurm stdout/stderr
$SCRATCH/mlfcs-gapa/tmp           temporary processing
```

Because scratch is purged, periodically copy important artifacts back to a durable location:

```bash
rsync -av --include='*/' --include='*.csv' --include='*.json' --include='*.md' --include='*.pt' --exclude='*' \
  "$SCRATCH/mlfcs-gapa/runs/" "$HOME/mlfcs-gapa-results/"
```

## Starter Batch Scripts

### CPU Smoke Test

```bash
#!/bin/bash
#SBATCH --job-name=mlfcs-smoke
#SBATCH --account=ls_math
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
cd "$SCRATCH/mlfcs-gapa"
source .venv/bin/activate

python - <<'PY'
import sys
import torch
print(sys.version)
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

Submit:

```bash
mkdir -p logs
sbatch smoke_cpu.sh
myjobs
```

### GPU Training Job

```bash
#!/bin/bash
#SBATCH --job-name=mlfcs-ppo
#SBATCH --account=ls_math
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --tmp=50G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
cd "$SCRATCH/mlfcs-gapa"
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

python -m mlfcs_gapa.train_ppo \
  --config configs/c_ppo.yaml \
  --output-dir runs/c_ppo_${SLURM_JOB_ID}
```

### CPU Data Build Job with Local Tmp

```bash
#!/bin/bash
#SBATCH --job-name=mlfcs-data
#SBATCH --account=ls_math
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=100G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
cd "$SCRATCH/mlfcs-gapa"
source .venv/bin/activate

rsync -a data/raw/ "$TMPDIR/raw/"

python -m mlfcs_gapa.data.build_lob \
  --input "$TMPDIR/raw" \
  --output "$TMPDIR/processed"

rsync -a "$TMPDIR/processed/" data/processed/
```

## Operational Checklist Before Running Real Experiments

1. Confirm current share:

```bash
my_share_info
```

2. Confirm quotas:

```bash
lquota
```

3. Confirm code location:

```bash
cd "$SCRATCH/mlfcs-gapa"
git status
```

4. Confirm Python:

```bash
module load stack/2024-06 python/3.12.8
source .venv/bin/activate
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

5. Submit a short smoke job before long jobs.

Current project smoke:

- job `67026477`
- date: 2026-05-18
- state: `COMPLETED`
- exit code: `0:0`
- elapsed: `00:02:59`
- max RSS: about `1.2 GB`
- covered pytest, synthetic generation, 7 baseline methods, Attn-LOB pretraining checkpoint, C-PPO smoke, and D-DQN smoke.

Latest expanded smoke:

- job `25494`
- date: 2026-05-19
- state: `COMPLETED`
- exit code: `0:0`
- elapsed: `00:03:03`
- max RSS: about `1.1 GB`
- covered 51 tests, synthetic generation, 7 baseline methods, Attn-LOB pretraining checkpoint, C-PPO smoke, D-DQN smoke, latency metrics, summary metrics, latency figure, decision trace, and attention heatmap.
- sync note: use root-anchored rsync excludes such as `/data/`, `/runs/`, and `/logs/`. Do not use unanchored `data/`, because that also excludes `src/mlfcs_gapa/data/`.

6. Watch with:

```bash
myjobs
squeue -u "$USER"
tail -f logs/<job>.out
```

7. After completion:

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,MaxRSS
```

8. If a job fails:

```bash
cat logs/<job>.err
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

## Project-Specific Euler Strategy

Use a staged compute plan:

1. Local machine:
   - implement simulator
   - unit tests
   - tiny synthetic data
   - shape checks for Attn-LOB

2. Euler CPU jobs:
   - build synthetic/public LOB datasets
   - run fixed/random/AS/Inv-RL/LOB-RL baselines
   - run short RL smoke tests
   - current smoke script also runs tiny C-PPO and D-DQN train/eval checks

3. Euler GPU jobs:
   - Attn-LOB pretraining
   - C-PPO training
   - D-DQN training
   - never run more than 4 GPUs at once
   - use the provided Slurm arrays:
     - `scripts/euler/pretrain_table_gpu_array.sh` uses `#SBATCH --gpus=1` and `#SBATCH --array=0-3%4`
     - `scripts/euler/train_agents_gpu_array.sh` uses `#SBATCH --gpus=1` and `#SBATCH --array=0-1%4`
     - `scripts/euler/latency_agents_gpu_array.sh` uses `#SBATCH --gpus=1` and `#SBATCH --array=0-11%4`
     - `scripts/euler/ablation_agents_gpu_array.sh` uses `#SBATCH --gpus=1` and `#SBATCH --array=0-7%4`
   - `train_agents_gpu_array.sh` accepts ablation environment variables:
     - `LOB_MODE=attn|mlp|none`
     - `USE_DYNAMIC_STATE=true|false`
     - `USE_AGENT_STATE=true|false`

4. Euler CPU/GPU mixed:
   - latency sweeps
   - ablations
   - plotting and metrics aggregation
   - `scripts/euler/runtime_cpu.sh` runs the Table III-style runtime benchmark without GPUs

Keep every experiment output self-describing:

```text
runs/<method>/<dataset>/<timestamp-or-jobid>/
  config.yaml
  metrics.csv
  episode_metrics.parquet
  trades.parquet
  checkpoints/
  slurm_job.txt
  git_commit.txt
```

## Known Open Issues

- `uv` is not currently available as a command or module on Euler in this account. Decide whether to install `uv` under `$HOME` or use `python -m venv`.
- No `/cluster/project/ls_math` or `/cluster/work/ls_math` path was visible. Use personal scratch unless another group path is provided.
- Apptainer exists but may require access through `get-access`.
- The current local PyTorch lockfile may choose CPU wheels on macOS. On Euler GPU nodes we need to ensure CUDA-enabled PyTorch is installed.
- Scratch is purged; important trained models and metrics must be copied to durable storage.
- Earlier sync attempts on 2026-05-19 failed before job submission because SSH to `euler.ethz.ch:22` timed out. Connectivity later recovered and job `25494` passed.

## Latest MLFCS-GAPA Euler State, 2026-05-19

Successful jobs:

- `28271` CPU smoke: 51 tests passed plus synthetic smoke commands.
- `28596` GPU pretraining array: FC-LOB, Conv-LOB, DeepLOB, and Attn-LOB synthetic checkpoints/metrics.
- `30992` CPU runtime benchmark.
- `30994` GPU main agent array: C-PPO and D-DQN.
- `30996` GPU latency agent array: C-PPO and D-DQN across `[1, 5, 10, 20, 50, 100]`.
- `30999` GPU ablation array: all eight C-PPO/D-DQN variants completed.
- `35862` CPU latency baseline sweep: Fixed, Random, AS, Inv-RL, LOB-RL across the same latency grid.

Generated result artifacts under `${SCRATCH}/mlfcs-gapa/runs`:

- `combined_metrics_30994_30996_30999_35862.csv`
- `latency_metrics_30996_35862.csv`
- `ablation_summary_30999.csv`
- `summary_metrics_30994_30996_30999_35862.csv`
- `latency_figure_30996_35862.png`

Operational notes:

- The four-GPU cap was respected throughout with Slurm array caps of `%4`.
- A direct `sbatch --wrap` latency-baseline attempt failed as job `35624` because `/bin/sh` does not support `source`. Use a Bash script or `bash -lc`, and prefer `scripts/euler/latency_baselines_cpu.sh`.
- The next GPU run should be a C-PPO calibration job using explicit PPO hyperparameters and one GPU. Do not launch a full multi-seed sweep until C-PPO no longer collapses to max-spread/no-trade on synthetic data.
- C-PPO direct `[0, 1]` action calibration jobs `37107`, `37111`, `37114`, and `37118` all completed but still produced no fills. The next run should use the normalized PPO action mapping now available through `NORMALIZE_ACTIONS=true`, which maps external `[-1, 1]` PPO actions to internal paper `[0, 1]` actions.
- Normalized-action C-PPO jobs `38766` and `38797` also no-traded with default PPO exploration scale.
- Narrow-exploration C-PPO jobs:
  - `40846`, `PPO_LOG_STD_INIT=-1`: completed, 8 fills, PnL about `-1.0`.
  - `40849`, `PPO_LOG_STD_INIT=-2`: completed, 28 fills, PnL about `8.0`.
- Current best next C-PPO setting is `NORMALIZE_ACTIONS=true` with `PPO_LOG_STD_INIT=-2`. Launch longer or multi-seed C-PPO runs only after checking unrelated account jobs with `squeue -u "$USER"`; three non-MLFCS GPU jobs were active after `40849`.
