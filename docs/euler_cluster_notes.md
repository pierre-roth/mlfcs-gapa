# Euler Cluster Notes

Current status: 2026-06-11.

This note follows the official ETH Euler storage documentation:
https://docs.hpc.ethz.ch/hardware/storage/

## Storage Roles

Euler storage has different intended uses. Keep this project aligned with those
roles.

| Location | Role for this project |
| --- | --- |
| `$HOME` / `/cluster/home/piroth` | Private long-term small files. Keep the code clone under `$HOME/projects/mlfcs-gapa`. |
| `$SCRATCH` / `/cluster/scratch/piroth` | Temporary job working space only. Do not keep project code or durable outputs here. |
| `/cluster/project/math/piroth` | Long-term critical project storage. Use sparingly for small, important artifacts or metadata. |
| `/cluster/work/math/piroth` | High-performance shared storage for large project data and durable experiment outputs. |
| `$TMPDIR` | Node-local temporary storage inside a Slurm job. Slurm deletes it when the job ends. |

Official notes that matter here:

- Scratch is not backed up and is automatically cleaned after roughly two weeks.
- Project storage is for critical long-term group data and has snapshots/backups.
- Work storage is Lustre, optimized for I/O performance and large files, with
  periodic backups.
- `lquota` without arguments reports home and scratch. For project/work shares,
  run `lquota /cluster/project/math/piroth` or
  `lquota /cluster/work/math/piroth`.

## Project Layout

Use this layout from now on:

```text
$HOME/projects/mlfcs-gapa
  code clone, .venv, Slurm scripts, logs directory

/cluster/work/math/piroth/mlfcs-gapa/data
  retained real data

/cluster/work/math/piroth/mlfcs-gapa/data_20260330.tar.zst
/cluster/work/math/piroth/mlfcs-gapa/data_20260330.tar.zst.sha256
  retained data archive and checksum

/cluster/work/math/piroth/mlfcs-gapa/runs
  future durable Euler run outputs, created only when experiments are launched
```

The Euler scripts in `scripts/euler/` now default to `$HOME/projects/mlfcs-gapa`
as the code directory. Durable run outputs default to
`/cluster/work/math/piroth/mlfcs-gapa/runs/...`; smoke-test outputs default to
`$SCRATCH/mlfcs-gapa-smoke/...`.

## Cleanup Performed

The 2026-06-11 cleanup removed old project code clones, environments, stale run
outputs, and stale synthetic artifacts from Euler. The deletion pass used exact
project-specific paths only.

Removed:

- `/cluster/home/piroth/venvs/mlfcs-gapa-gpu`
- `/cluster/scratch/piroth/mlfcs-gapa`
- `/cluster/scratch/piroth/mlfcs-gapa-throughput`
- `/cluster/scratch/piroth/venvs/mlfcs-gapa-euler-py311`
- `/cluster/project/math/piroth/mlfcs-gapa`
- `/cluster/work/math/piroth/mlfcs-gapa/profile_base1`

Kept:

- `/cluster/home/piroth/public/mlfcs-gapa`
  - `data_20260330.tar.zst`
  - `data_20260330.tar.zst.sha256`
  - `README.txt`
- `/cluster/work/math/piroth/mlfcs-gapa`
  - `data/`
  - `data_20260330.tar.zst`
  - `data_20260330.tar.zst.sha256`

The live verification after cleanup found only these remaining `mlfcs-gapa`
locations:

- `/cluster/home/piroth/public/mlfcs-gapa`
- `/cluster/work/math/piroth/mlfcs-gapa`

## Job Hygiene

Before submitting jobs:

```bash
ssh euler
cd "$HOME/projects/mlfcs-gapa"
mkdir -p logs
bash scripts/euler/setup_venv.sh
```

For job status:

```bash
squeue -u "$USER"
```

For W&B tracking:

```bash
export WANDB_ENABLED=true
export WANDB_ENTITY=piroth-ethz
export WANDB_PROJECT=mm-drl-lob
# Optional for dry runs or no network:
# export WANDB_MODE=offline
```

The job scripts source `scripts/euler/wandb_env.sh`. Tracking is off unless
`WANDB_ENABLED` is set to `true`, `1`, `yes`, or `on`.

For storage usage:

```bash
lquota
lquota /cluster/project/math/piroth
lquota /cluster/work/math/piroth
```

Do not clean paths by generic names such as `logs`, `runs`, or `venv` outside an
identified `mlfcs-gapa` path. Other projects are present on the account and must
not be touched.
