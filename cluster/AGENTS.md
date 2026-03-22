# AGENTS.md — Working effectively on the ETH Zürich Euler cluster

This document is for an autonomous coding / research agent operating on the ETH Zürich Euler HPC cluster on behalf of user **`piroth`**.

It is written to make the agent productive **without violating cluster policy, overloading shared resources, or making bad assumptions about Euler-specific behavior**.

---

## 1. Mission and operating model

Euler is **ETH Zürich’s central HPC cluster**. It is a **shared**, **heterogeneous**, **Slurm-managed** system intended for scientific and engineering workloads. Jobs run on compute nodes; login nodes are for light interactive work such as editing files, submitting jobs, inspecting results, and debugging small issues. citeturn845255view0turn845255view1turn845255view2turn845255view7

Treat Euler as a production HPC environment, not as a personal VM.

Your operating priorities, in order:

1. **Respect cluster policy and shared usage.**
2. **Preserve user data.**
3. **Keep runs reproducible.**
4. **Minimize queue time and wasted allocation.**
5. **Leave behind clear logs, scripts, and outputs.**

---

## 2. Identity and user-specific assumptions

The ETH username is **`piroth`**.

Use these paths as defaults unless inspection shows otherwise:

- Home: `/cluster/home/piroth`
- Scratch: `/cluster/scratch/piroth`
- Likely scratch env var: `$SCRATCH`
- Slurm local scratch during a job: `$TMPDIR`

Do **not** assume `piroth` has GPU access through a shareholder group. On Euler, the public share has **no GPUs** and allows up to **48 CPU cores** and **128 GiB RAM**. GPU access normally requires shareholder-backed GPU resources. citeturn845255view10

Because group membership is not known in advance, probe the environment before choosing a plan.

Recommended first checks:

```bash
whoami
pwd
lquota
id
module avail 2>&1 | head -n 50
```

Interpretation:

- `whoami` should return `piroth`.
- `pwd` should normally begin in `/cluster/home/piroth`.
- `lquota` shows available home/scratch and possibly group storage quotas.
- `id` may reveal shareholder or special-access groups.

---

## 3. Hard rules: what you must and must not do

### 3.1 Never run heavy workloads on login nodes

Euler has **50 login nodes**, each with **32 GB RAM**, intended for SSH access, file management, compilation, job submission, and light debugging. They are **not** for long-running or resource-intensive computations. citeturn845255view7

Allowed on login nodes:

- editing files
- git operations
- lightweight Python/bash inspection
- compilation
- preparing job scripts
- monitoring jobs
- tiny smoke tests that finish quickly and use negligible resources

Not allowed on login nodes:

- training models
- long simulations
- multithreaded heavy preprocessing
- memory-hungry data transforms
- persistent services unless explicitly supported via Euler services

If a command might use meaningful CPU, memory, or runtime, move it into a Slurm allocation.

### 3.2 Prefer Slurm for anything nontrivial

Euler uses **Slurm 25.05** and runs jobs **non-exclusively** on compute nodes by default. Request only what is needed. Smaller, accurate requests start sooner. Do **not** specify partitions unless absolutely necessary. citeturn845255view2turn504974view0

### 3.3 Do not assume hardware homogeneity

Euler is heterogeneous. CPU type, RAM, local disk, network, and GPU model can vary by node generation. Avoid overfitting scripts to a specific node type unless the workload truly depends on it. citeturn845255view1turn845255view6

### 3.4 Protect data placement

- Permanent/important data belongs in **home**, **project**, or **work**.
- Large temporary working sets belong in **scratch** or job-local **`$TMPDIR`**.
- Never rely on scratch for preservation.

Scratch files older than roughly **15 days / 2 weeks** are deleted automatically, and scratch has **no backup**. citeturn504974view8turn504974view9

---

## 4. Euler filesystem model: where to put things

### 4.1 Home

Use home for:

- source code
- git repos
- small configs
- job scripts
- lightweight results
- irreplaceable personal files

Home is a long-lived filesystem with snapshots and backups. The table in ETH docs lists **50 GB per user** and snapshots/backups for home. citeturn504974view9

Suggested layout:

```text
/cluster/home/piroth/
  projects/
  src/
  jobs/
  logs/
  envs/
  bin/
  tmp-small/
```

### 4.2 Scratch

Use scratch for:

- large intermediate datasets
- unpacked archives
- training checkpoints you can regenerate
- temporary conda/apptainer caches
- stage-in/stage-out workflows

Important Euler-specific facts:

- path: `/cluster/scratch/piroth`
- limit: **2.5 TB** and **1,000,000 files/directories**
- automatic deletion after **2 weeks**
- no snapshots, no backups citeturn504974view8turn504974view9

Scratch is optimized for **large files**. Avoid huge trees of tiny files.

### 4.3 Node-local scratch (`$TMPDIR`)

During a Slurm job, request local scratch with `--tmp=<size>` and use `$TMPDIR` for fast local I/O. ETH explicitly documents that Slurm creates a unique per-job local scratch directory and deletes it automatically after the job finishes. citeturn504974view8

This is often the best place for:

- decompression of many small files
- sort/merge temp files
- databases created on the fly
- PyTorch/JAX temporary artifacts
- container temporary files

Pattern:

1. copy input from home/scratch to `$TMPDIR`
2. run compute there
3. copy outputs back before exit

---

## 5. First-session discovery checklist

At the start of a task or after a fresh login, do this:

```bash
whoami
hostname
pwd
lquota
id
module avail 2>/dev/null | head -n 100
module spider python 2>/dev/null | head -n 100
sinfo -o "%P %D %c %m %G %f"
```

Then classify the situation:

- **CPU-only public-share workflow**: safest default
- **group-backed CPU workflow**: possibly larger resources available
- **GPU-capable workflow**: only if access is confirmed or a special documented path applies

Also check current system health when behavior looks strange. ETH publishes Euler system status, and as of 22 March 2026 the status page reports Euler operational. citeturn364724view0

---

## 6. Slurm usage on Euler

### 6.1 General policy

Euler guidance is explicit:

- request CPUs, memory, and walltime **accurately**
- keep requests **as small as practical**
- **ignore partitions** unless truly needed citeturn504974view0

Useful defaults from the docs:

- default runtime: **1 hour**
- default tasks: **1**
- default CPUs per task: **1** citeturn504974view5

### 6.2 Interactive sessions

For debugging or manual runs:

```bash
srun --pty bash
```

ETH documents this exact pattern for interactive sessions. citeturn504974view1

Add resources when needed:

```bash
srun --time=02:00:00 --cpus-per-task=8 --mem-per-cpu=4G --pty bash
```

Use interactive jobs for:

- debugging builds
- running notebooks through supported services
- inspecting performance
- launching short GUI/X11 workflows if supported

### 6.3 Batch jobs

Canonical structure:

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
# module load ...

srun ./run.sh
```

Use `srun` inside the batch script. Euler’s docs warn that without `srun`, a multi-task batch job will not execute across all requested tasks as expected. citeturn504974view2

### 6.4 Monitoring

Common commands:

```bash
squeue -u piroth
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem,AllocCPUS
scancel <jobid>
```

ETH also highlights `myjobs` in its first-job tutorial as a convenience command for monitoring. citeturn395284search2

### 6.5 How to size requests

Heuristics for the agent:

- Start with **1 task** unless using MPI or embarrassingly parallel `srun`/job arrays.
- For threaded codes, use `--ntasks=1 --cpus-per-task=N`.
- For MPI, use `--ntasks=N` and only add `--cpus-per-task` when each rank is multithreaded.
- Set walltime based on a measured pilot run, then add a moderate safety margin.
- Prefer `--mem-per-cpu` unless you have a strong reason to use another memory expression.
- Avoid oversizing “just in case”. That slows scheduling and wastes shared resources.

### 6.6 Public-share safe ceiling

If you do not know whether `piroth` is attached to a shareholder group, assume public-share constraints and stay within:

- at most **48 CPU cores**
- at most **128 GiB RAM**
- **no standard GPU access** citeturn845255view10

---

## 7. GPU policy and GPU workflows

### 7.1 Standard Euler GPU access

ETH’s docs state that GPU nodes are only available to **shareholders** that invest in GPU hardware. Do not assume general GPU access. citeturn845255view1turn504974view4

If access is confirmed, the documented Slurm pattern is:

```bash
srun --gpus=1 nvidia-smi
```

or specify a GPU type only when the workload genuinely requires it. citeturn504974view4

### 7.2 MI300A evaluation path

ETH currently documents a special **AMD MI300A evaluation** with broader access: anyone with an ETH account has access to at least one MI300A APU, using a **dedicated Slurm instance** selected by setting:

```bash
export SLURM_CONF=/cluster/adm/slurm-amdgpu/slurm/etc/slurm.conf
```

The docs note that memory requests must account for both CPU and GPU memory. Example interactive request:

```bash
srun --ntasks=1 --cpus-per-task=24 --time=4:00:00 --gpus=mi300a:1 --mem-per-cpu=4G --pty bash
```

This is current, but special-purpose. Use it only when AMD GPU testing is intentional. citeturn845255view11turn395284search9

### 7.3 GPU agent behavior

Decision order:

1. Prefer CPU unless the task clearly benefits from GPUs.
2. Check whether standard GPU access exists.
3. If not, consider the MI300A evaluation path only if the software stack supports AMD/ROCm and the task is suitable.
4. Never silently change Slurm instances without recording it in logs/job scripts.

---

## 8. Software environment on Euler

### 8.1 Modules and software stacks

Euler uses **Lua-based environment modules** arranged in layered stacks. Current named stacks visible in the docs include:

- `stack/2025-06`
- `stack/2024-06`
- `stack/2024-04` citeturn845255view4turn504974view7

The **2025-06** stack is based on **Spack 0.23.1**, deployed under `/cluster/software/stacks/2025-06`, and includes over **1,800** libraries/applications plus multiple compiler families. citeturn504974view7

Practical rules:

- Start scripts with `module purge` when reproducibility matters.
- Explicitly load a stack before loading software if required.
- Use `module spider <name>` when `module avail` is insufficient.

ETH’s FAQ explicitly recommends `module avail` and `module spider` and notes that some modules are hidden until prerequisites are loaded. citeturn395284search7

### 8.2 Python

ETH provides centrally installed Python modules. The current Python page lists, for example, Python 3.12.8 on `stack/2024-06`, plus CUDA-enabled Python variants. citeturn395284search3

Typical safe pattern:

```bash
module purge
module load stack/2024-06 python/3.12.8
python -V
```

Before inventing a custom environment, check whether required packages are already present.

### 8.3 Conda guidance

ETH explicitly says there is **no centrally installed Conda**. Users may install it themselves, but Euler’s Lustre filesystem is **not well suited** to Conda because Conda creates many small files, which can hurt performance and quotas. citeturn395284search0

Agent policy:

- Prefer centrally installed modules first.
- Prefer **venv on top of module Python** when feasible.
- Use Conda only when necessary.
- If using Conda/Mamba, place caches and environments thoughtfully to reduce metadata pressure.
- Do not explode giant package trees in home without need.

### 8.4 Spack

ETH documents Spack usage and activation via stack setup scripts, e.g.:

```bash
. /cluster/software/stacks/2025-06/setup-env.sh
```

Use this when you need package builds that align with the cluster’s provided stack. citeturn149649search18turn504974view7

### 8.5 Containers via Apptainer

Apptainer is available on Euler as an OS package; **no module is needed**. Access is restricted; users request it by running:

```bash
get-access
```

ETH recommends:

```bash
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer"
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}"
```

ETH also states that building custom containers is **not permitted directly on Euler**; build them elsewhere and transfer the `.sif` file to Euler. GPU use is via `--nv`. citeturn362870view0

Agent policy for Apptainer:

- Check access first:

```bash
id | grep ID-HPC-SINGULARITY
```

- Put cache in scratch.
- Put temp build/runtime files in local scratch when possible.
- Bind only the directories you need.
- For GPU containers, use `--nv` only inside a GPU-capable allocation.

---

## 9. Data movement and remote tooling

### 9.1 SSH access

Euler login entry point:

```bash
ssh piroth@euler.ethz.ch
```

This is the documented hostname. A load balancer distributes sessions across 50 login nodes. Specific login node hostnames also exist, but normally you should use `euler.ethz.ch`. citeturn845255view7turn741974search15

### 9.2 Euler tunnel for compute-node access

ETH provides **`euler-tunnel`** to establish SSH tunnels directly to running batch jobs. This is the recommended path for workflows like VS Code Remote SSH against a compute-node job instead of a login node. citeturn741974search0turn741974search9

This is useful when the agent needs:

- long-lived terminal sessions on compute nodes
- remote IDE attachment to compute allocations
- stable reconnects that avoid login-node load-balancer issues

### 9.3 Web services

ETH documents these Euler services:

- **JupyterHub**
- **RStudio**
- **TensorBoard**
- **code-server / VS Code in browser** citeturn741974search7turn364724view0turn741974search2

Use them when the workflow benefits from managed interactive sessions instead of ad hoc background services.

### 9.4 Globus for large transfers

ETH recommends Globus for efficient large-scale data movement between Euler and other systems. For large transfers, prefer Globus over improvised fragile copy loops. citeturn741974search3turn364724view0

---

## 10. Reproducibility standards for this agent

For every substantial run, leave behind:

- the exact job script
- the exact git commit or source snapshot
- loaded modules
- important environment variables
- job ID
- start/end timestamps
- stdout/stderr logs
- key output manifest

Minimum boilerplate to emit at job start:

```bash
echo "DATE=$(date -Is)"
echo "HOST=$(hostname)"
echo "USER=$(whoami)"
echo "PWD=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
module list 2>&1 || true
env | sort > "env.${SLURM_JOB_ID:-manual}.txt"
```

Prefer directory names like:

```text
runs/YYYY-MM-DD_taskname/
```

and inside:

```text
job.sbatch
stdout.txt
stderr.txt
env.txt
metadata.json
artifacts/
```

---

## 11. Safe job templates

### 11.1 CPU batch template

```bash
#!/bin/bash
#SBATCH --job-name=cpu-task
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 077

mkdir -p logs results

module purge
module load stack/2024-06 python/3.12.8

python -V

srun python script.py --input data/input.dat --output results/output.dat
```

### 11.2 Local-scratch-heavy template

```bash
#!/bin/bash
#SBATCH --job-name=tmpdir-task
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=100G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
umask 077

module purge

mkdir -p logs results
rsync -a input/ "$TMPDIR/input/"
cd "$TMPDIR"

run_my_pipeline input results

rsync -a "$TMPDIR/results/" "$SLURM_SUBMIT_DIR/results/"
```

This mirrors ETH’s documented `$TMPDIR` workflow. citeturn504974view8

### 11.3 Interactive debug template

```bash
srun --time=01:00:00 --cpus-per-task=4 --mem-per-cpu=4G --pty bash
```

Then inside:

```bash
module purge
module load stack/2024-06 python/3.12.8
python -V
```

### 11.4 GPU template only if access is confirmed

```bash
#!/bin/bash
#SBATCH --job-name=gpu-task
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module purge
# load modules required by your GPU stack
srun nvidia-smi
srun python train.py
```

---

## 12. Resource-tuning heuristics

### 12.1 CPU-bound code

Use:

- `--ntasks=1`
- `--cpus-per-task=N`

Also set thread-count env vars when relevant:

```bash
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
```

### 12.2 MPI code

Use multiple tasks and keep topology explicit. Start from a small scale, validate correctness, then scale out.

### 12.3 Python scientific stack

Default stance:

- start single-process
- constrain BLAS/OpenMP threads deliberately
- avoid accidental oversubscription
- prefer batch jobs for long pandas/NumPy/scikit-learn runs

### 12.4 Many-small-files workloads

Euler scratch and especially Conda-like trees can be painful due to metadata-heavy access patterns. Prefer:

- tar/untar once into `$TMPDIR`
- databases / parquet / sharded archives over millions of tiny files
- per-job local workspaces

This aligns with ETH’s warnings about scratch and Conda behavior on Lustre-like shared storage. citeturn504974view8turn395284search0

---

## 13. Failure handling and troubleshooting

When a job fails, follow this order:

1. Read stdout/stderr.
2. Inspect `sacct` for state, elapsed time, and memory use.
3. Reproduce interactively with a small allocation if needed.
4. Check module environment drift.
5. Check data paths and quotas with `lquota`.
6. Check whether scratch files expired or were written to the wrong place.
7. Check system status / recent maintenance announcements. citeturn364724view0turn207175search10

Common pitfalls on Euler:

- running compute on a login node instead of in Slurm
- using scratch for important outputs
- forgetting `srun` inside a multi-task batch script
- over-requesting resources and waiting forever
- relying on hidden modules without loading prerequisites
- Conda environments causing metadata or quota problems
- trying to build Apptainer images directly on Euler

When stuck, ETH support channels are explicit:

- `cluster-support@id.ethz.ch`
- SmartDesk / Service Desk
- in-person Euler help desk sessions every two weeks in 2026 citeturn741974search4turn845255view9turn364724view0

---

## 14. Security, privacy, and etiquette

- Use restrictive file permissions for sensitive work: `umask 077` when appropriate.
- Never hard-code ETH credentials.
- Do not expose tokens in logs.
- Clean up scratch and `$TMPDIR` artifacts promptly.
- Avoid busy polling loops against Slurm commands.
- Do not launch persistent servers on login nodes.
- Avoid abusive parallel file operations on shared storage.

ETH explicitly ties first login to the cluster usage rules and ETH acceptable-use policy. citeturn741974search15

---

## 15. Decision tree for the agent

### If the task is small and exploratory

- work in home
- run quick inspection on login node
- move to `srun --pty bash` as soon as compute becomes nontrivial

### If the task is CPU-heavy

- assume CPU-only public-share mode unless stronger access is known
- submit a batch job
- use `$TMPDIR` for temp-heavy work

### If the task needs many packages

- first check module stack
- then venv on top of module Python
- only then consider Conda
- consider Apptainer if software is hard to reproduce otherwise

### If the task needs a GPU

- verify access first
- otherwise consider whether the MI300A evaluation path is appropriate
- record the exact Slurm/GPU environment used

### If the task needs remote IDE / notebook workflow

- prefer JupyterHub / code-server / euler-tunnel
- do not anchor workflows to login nodes

---

## 16. Recommended command snippets for `piroth`

### Basic access

```bash
ssh piroth@euler.ethz.ch
```

### Inspect quotas and storage

```bash
lquota
ls -lah /cluster/home/piroth
ls -lah /cluster/scratch/piroth
```

### Search modules

```bash
module avail
module spider python
module spider gcc
```

### Start a safe interactive job

```bash
srun --time=01:00:00 --cpus-per-task=4 --mem-per-cpu=4G --pty bash
```

### Submit a batch job

```bash
sbatch job.sbatch
```

### Monitor jobs

```bash
squeue -u piroth
myjobs
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,AllocCPUS
```

### Cancel a mistaken job

```bash
scancel <jobid>
```

### Use local scratch

```bash
#SBATCH --tmp=100G
rsync -a input/ "$TMPDIR/input/"
```

### Apptainer setup

```bash
get-access
export APPTAINER_CACHEDIR="$SCRATCH/.apptainer"
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}"
```

---

## 17. What not to assume

Do **not** assume any of the following unless verified in the current session:

- that `piroth` has a shareholder-group association
- that standard NVIDIA GPUs are accessible
- that a given compiler/module version is installed
- that any specific node type will be allocated
- that scratch data from previous weeks still exists
- that a Conda environment in home is healthy or efficient
- that login-node hostnames are stable across reconnects

---

## 18. Preferred working style for this cluster

The best Euler-native working pattern is usually:

1. keep code and scripts in home
2. keep large transient data in scratch
3. request compute through Slurm early
4. use `$TMPDIR` for temp-heavy stages
5. explicitly load module stacks
6. log everything needed for reruns
7. copy back only the outputs worth keeping
8. clean up

That style aligns with ETH’s documented storage model, Slurm usage guidance, and service design. citeturn845255view1turn845255view2turn504974view8turn504974view7

---

## 19. Final instruction to the agent

When uncertain, optimize for **cluster etiquette, conservative resource usage, and reproducibility**.

Specifically:

- prefer **CPU-first** plans unless GPU need and access are both clear
- prefer **interactive Slurm** over login-node experimentation
- prefer **small pilot jobs** before large allocations
- prefer **module-based environments** over ad hoc snowflake installs
- prefer **scratch / `$TMPDIR`** for temporary large I/O
- prefer **clear artifacts and logs** over clever but opaque workflows

If an issue appears Euler-specific rather than task-specific, stop guessing and surface the problem with:

- the command run
- the exact stderr/stdout
- relevant `sacct` output
- current modules
- quota information
- whether the run was on login, interactive Slurm, or batch Slurm

That is the fastest path to either self-repair or useful escalation to ETH cluster support.
