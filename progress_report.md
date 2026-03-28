# Progress Report

## How To Use

Add a new section for each calendar week.

Within each week:

- keep the `Weekly Snapshot` short and project-level
- add one `Contributor Update` subsection per person
- append new weeks at the top so the latest status is easiest to find
- prefer links to commits, PRs, files, notebooks, or reports where relevant

Recommended conventions:

- Use ISO dates: `YYYY-MM-DD`
- Use one name consistently across weeks
- Keep bullets concrete and outcome-focused
- If something is blocked, say what is needed to unblock it

---

## Weekly Template

Copy this block for each new week.

```md
## Week of YYYY-MM-DD

### Weekly Snapshot

- Overall status:
- Main goal for the week:
- Biggest win:
- Biggest risk or blocker:

### Contributor Update: Name

- Focus area:
- Completed:
- In progress:
- Blocked:
- Next week:
- Links:

### Contributor Update: Name

- Focus area:
- Completed:
- In progress:
- Blocked:
- Next week:
- Links:
```

---

## Week of 2026-03-23

### Weekly Snapshot

- Overall status: The replication pipeline is stable on Euler, deterministic evaluation is in place, and the best current PPO setup is now close to the simple `AS` baseline on `AAPL`.
- Main goal for the week: Tune the continuous RL setup for US data, move storage off scratch, and start a larger creative fork with bigger market-structure changes.
- Biggest win: The stage-6 competitive quoting setup materially improved PPO and identified a credible default for larger runs.
- Biggest risk or blocker: PPO still learns weak directional bias and the current `lobmm` environment likely remains mismatched to US microstructure.

### Contributor Update: Pierre

- Focus area: Euler training pipeline, RL tuning, environment realism, and experimental infrastructure.
- Completed:
  - Removed real-run downsampling by default, made the encoder trainable by default, aligned smoke/full more closely, and improved US-specific defaults in `lobmm/`.
  - Simplified the baseline suite to primitive baselines, moved Euler data/artifacts/logs to permanent `work`/`project` storage, and updated the cluster defaults accordingly.
  - Added resumable pretraining, deterministic evaluation, PPO checkpoint plumbing, policy diagnostics, and multiple Euler submission helpers.
  - Ran the full tuning sequence on Euler from `medium` and `full` runs through stage-2/3/4/5/6 AAPL sweeps, then launched two larger stage-7 full runs and a new creative `lobmmx/` fork.
  - Created `lobmmx/` as a sibling experimental package with random initial inventory, terminal inventory allowed, trading-edge reward, spread/tick-unit reward scaling, US-timescale features, decoupled directional vs inventory skew, multitask pretraining, maker/taker fees, and aggressive validation-time PPO checkpointing.
- In progress:
  - Stage-7 full runs on `AAPL+GOOGL` are running on Euler with the current best `lobmm` setup.
  - The first `lobmmx` creative AAPL batch is running on Euler.
- Blocked:
  - Current PPO still tends toward low directional bias, so further gains likely require environment/state changes rather than more standard hyperparameter tuning.
- Next week:
  - Compare the stage-7 full runs and the first `lobmmx` runs.
  - Keep the best current-code setup as the exploitation path and iterate on `lobmmx` as the exploration path.
- Relevant findings from runs:
  - `euler_aapl_medium`: pipeline healthy; end-to-end runtime about `23.5m`.
  - `euler_full_12h`: first full run failed in PPO due to CUDA OOM; fixed by moving only minibatches to GPU.
  - `euler_full_tuned_24h`: stronger budget improved pretrain (`AAPL F1 0.687`, `GOOGL F1 0.610`) and made `GOOGL` slightly profitable, but still far from the paper.
  - Stage-2 AAPL sweep: balanced pretraining solved the class-collapse issue; `ctrl`/`sampler` reached about `0.64` val/test F1, but PPO still trailed `AS`.
  - Stage-3 AAPL sweep: better pretraining stayed stable, but PPO remained too passive; lowering `zeta` or freezing the backbone had little effect.
  - Stage-4 AAPL sweep: `pnl_inventory` reward with `60s` episodes and much higher `gamma`/`GAE` worked best; `euler_aapl_stage4_pnlinv60_ultra` reached `pnl 0.01469`, `nd_pnl 0.13978`, `sharpe 0.3014`, close to `AS`.
  - Stage-5 AAPL sweep: larger RL budget and extra reward shaping hurt; competitive quotes were the only promising change.
  - Stage-6 AAPL sweep: stage-4-sized budget plus competitive quotes was the best current result; `euler_aapl_stage6_ultra_competitive_ckpt` reached `pnl 0.01406`, `nd_pnl 0.14454`, `sharpe 0.2576`, while the plain control lagged clearly.
  - Global conclusion so far: competitive quote scales matter a lot, deterministic evaluation was necessary, pretraining is now good enough, and the remaining ceiling is probably environment/state realism rather than more generic PPO tuning.
- Links:
  - `lobmm/`
  - `lobmmx/`
  - `cluster/`
  - `/cluster/project/math/piroth/mlfcs-gapa/artifacts/`
  - `/cluster/work/math/piroth/mlfcs-gapa/data/processed/`

## Week of 2026-03-16

### Weekly Snapshot

- Overall status: Data preprocessing and validation workflow is now in place for Databento-based experiments.
- Main goal for the week: Convert, validate, and package usable project data so method work can proceed on top of a stable input pipeline.
- Biggest win: The converter was validated against `MBP-10`, a major reconstruction bug was fixed, and a tracked sample dataset was added for contributors without the full raw data.
- Biggest risk or blocker: The legacy training stack still needs a fully resolved `uv` dependency story, especially around `tensorflow` and `tensorforce`.
- Added complete ground-up implemetation of the paper's continuous part in `lobmm/`.

### Contributor Update: Pierre

- Focus area: Data pipeline, project setup, and reproducibility.
- Completed:
  - Added Databento conversion, validation, and divergence-analysis tooling.
  - Converted and validated `GOOGL` and `AAPL`.
  - Added a tracked sample dataset in `data/sample`.
  - Moved the repo toward a `uv`-first setup and removed the old conda environment file.
- In progress:
  - Finalizing the runtime dependency story for the legacy training stack under `uv`.
- Blocked:
  - Full dependency resolution is constrained by older `tensorflow` and `tensorforce` compatibility.
- Next week:
  - Smoke-test the main project code against the converted dataset.
  - Continue closing the paper-to-code gaps in training and evaluation.
- Links:
  - `preprocessing/databento/`
  - `data/sample/`
  - `data/validation/`

# Contributor Update: Anja
- Wrote full simplified pipeline in `paper_replication.ipynb`

# Contributor Update: Pierre
- uploaded all data to cluster (currently in scratch as no dedicated storage found)
- Started writing complete paper (continuous part only) replication pipeline in PyTorch in `lobmm/`
- Warning: data normalization is still different than in the paper. (log instead of divide by max volume, no perfect stationarity, but assume paper doesn't have it either)


## Week of 2026-03-16

### Weekly Snapshot

TODO
