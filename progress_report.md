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

## Week of 2026-04-14

### Weekly Snapshot

- Overall status: selective branch integration worked mechanically, but only the reporting changes are clear keepers. The new opt-in queue-aware `lobmmx` fill model did not produce a better default than the current legacy fill model.
- Main goal for the week: fast-forward to the shared mainline, import safe improvements from side branches, and validate them on Euler with small AAPL-only runs before considering larger sweeps.
- Biggest win: annualized Sharpe reporting and a reusable fill-model validation workflow are now in place, and the branch integrations were tested end-to-end on Euler without breaking the pipeline.
- Biggest risk or blocker: the queue-aware fill variants changed trading behavior substantially but did not improve the overall policy in a clean way; `queue_back` was too pessimistic and `queue_uniform` increased churn and terminal-inventory penalties too much.

### Contributor Update: Pierre

- Focus area: selective branch integration, validation infrastructure, and AAPL-only `lobmmx` fill-model experiments.
- Completed:
  - Fast-forwarded `mm-drl-lob` to include the merged `lobmmx` reward-fix PR.
  - Imported annualized Sharpe reporting into both `lobmm/` and `lobmmx/`.
  - Added an opt-in queue-aware fill model to `lobmmx`, but kept `fill_model="legacy"` as the default so behavior does not silently change.
  - Added `diag_microstructure.py` as a small reusable diagnostic without merging the hard-coded GOOGL quote override.
  - Added `cluster/submit_lobmmx_fillmodel_validation.sh` and ran three AAPL-only Euler validation pipelines against the shared stage-2 pretrain:
    - `euler_lobmmx_stage3_legacy_control`
    - `euler_lobmmx_stage3_queue_back`
    - `euler_lobmmx_stage3_queue_uniform`
- Results:
  - `legacy_control` PPO: `pnl 0.01185`, `nd_pnl 0.13772`, `sharpe 0.526`, `reward -0.293`, `trades 1.19`
  - `queue_back` PPO: `pnl -0.00511`, `nd_pnl -0.05741`, `sharpe -0.0668`, `reward -0.933`, `trades 2.13`
  - `queue_uniform` PPO: `pnl 0.02353`, `nd_pnl 0.28562`, `sharpe 0.271`, `reward -6.161`, `trades 13.13`
- Conclusion:
  - The reporting changes helped and should stay.
  - The queue fill model does not justify becoming the new default.
  - `queue_back` is a clear regression.
  - `queue_uniform` improves raw PnL but does so by trading much more aggressively, greatly increasing turnover, position size, and terminal-inventory penalties; under the current reward design, this is not an unambiguous improvement.
  - Keep `fill_model="legacy"` as default for now and treat queue-aware fills as an experimental branch path that still needs reward/penalty retuning.
- Links:
  - `lobmm/`
  - `lobmmx/`
  - `cluster/submit_lobmmx_fillmodel_validation.sh`
  - `/cluster/project/math/piroth/mlfcs-gapa/artifacts/euler_lobmmx_stage3_legacy_control/`
  - `/cluster/project/math/piroth/mlfcs-gapa/artifacts/euler_lobmmx_stage3_queue_back/`
  - `/cluster/project/math/piroth/mlfcs-gapa/artifacts/euler_lobmmx_stage3_queue_uniform/`

## Week of 2026-03-30

### Weekly Snapshot

- Overall status: `lobmmx` reward distortion fixed; PPO now trains to a profitable policy on AAPL.
- Biggest win: Identified and fixed the terminal inventory penalty bug in `lobmmx` — terminal penalty was 10-80× larger than trading edge, making the reward signal uninformative.

### Contributor Update: Amine

- Focus area: `lobmmx` reward distortion diagnosis and fix.
- Diagnosed the terminal inventory penalty bug: penalty was computed on `abs(inventory)` instead of `abs(inventory - initial_inventory)`, causing agents with random initial inventory to receive strongly negative reward regardless of trading quality.
- Fixed `_terminal_inventory_penalty()` in `lobmmx/env.py` to use net inventory change from episode start.
- Added `diag_reward.py` diagnostic script to quantify reward components without running full training.
- Validated fix on Euler (medium mode, CPU): PPO now converges to `pnl_mean=0.026`, `nd_pnl_mean=0.229`, `sharpe=0.827` vs completely broken reward signal before.
- PR: https://github.com/pierre-roth/mlfcs-gapa/pull/1

---

## Week of 2026-03-23

### Weekly Snapshot

- Overall status: Mainline `lobmm` is stable but near a local ceiling on `AAPL`. The corrected `lobmmx` fork is competitive on `AAPL` in one variant, but its reward scale is still not coherent. Cluster data is now packaged for external sharing.
- Main goal for the week: Use the finished stage-8 and `lobmmx` stage-2 results to decide the next AAPL-focused experiments, and finish a clean data handoff path for collaborators.
- Biggest win: `lobmmx` `spread_aggr` nearly matched the mainline `AAPL` PPO result while learning materially larger directional bias, and the full cluster data was compressed from `84G` raw to a `12G` shareable archive.
- Biggest risk or blocker: `lobmmx` still assigns strongly negative reward to profitable policies, and the live `work` data path itself cannot be made directly team-readable because the parent namespace is private.

### Contributor Update: Pierre

- Focus area: Euler training pipeline, RL tuning, environment realism, and experimental infrastructure.
- Completed:
  - Removed real-run downsampling by default, made the encoder trainable by default, aligned smoke/full more closely, and improved US-specific defaults in `lobmm/`.
  - Simplified the baseline suite to primitive baselines, moved Euler data/artifacts/logs to permanent `work`/`project` storage, and updated the cluster defaults accordingly.
  - Added resumable pretraining, deterministic evaluation, PPO checkpoint plumbing, policy diagnostics, and multiple Euler submission helpers.
  - Ran two larger stage-7 full runs on `AAPL+GOOGL` using the best current mainline competitive configuration.
  - Created and exercised `lobmmx/` as a sibling experimental package with random initial inventory, terminal inventory allowed, trading-edge reward, spread/tick-unit reward scaling, US-timescale features, decoupled directional vs inventory skew, multitask pretraining, maker/taker fees, and aggressive validation-time PPO checkpointing.
  - Fixed the remaining `lobmmx` objective issues by switching PPO selection to `pnl_mean`, removing the default per-step inventory punishment, and replacing it with an explicit terminal liquidation-cost penalty.
  - Fixed the local smoke verification path so tests run from the tracked sample dataset when the full local processed dataset is absent.
  - Submitted and completed a corrected stage-2 `lobmmx` AAPL sweep and new large mainline stage-8 full runs.
  - Packaged the canonical cluster dataset into a compressed archive for collaborator handoff:
    - source live data remains at `/cluster/work/math/piroth/mlfcs-gapa/data`
    - backup/share archive is `/cluster/work/math/piroth/mlfcs-gapa/data_20260330.tar.zst`
    - public teammate-accessible copy is `/cluster/home/piroth/public/mlfcs-gapa/data_20260330.tar.zst`
    - checksum and README were added alongside it
  - Removed the old uncompressed backup after the compressed archive was created and copied successfully.
- In progress:
  - No active runs. Next experiments are being selected from the finished stage-8 and `lobmmx` stage-2 evidence.
  - A local download of the `12G` archive to the laptop `Downloads` folder is in progress for external file-sharing upload.
- Blocked:
  - Mainline `lobmm` still trails simple baselines on `AAPL`.
  - `lobmmx` still optimizes a distorted reward scale, so checkpoint selection is not yet trustworthy there.
- Next week:
  - Decide the next AAPL-focused runs from the finished results.
  - Prioritize reward-scale calibration and selection logic in `lobmmx`, plus any high-upside AAPL experiments worth a larger budget.
- Relevant findings from runs:
  - `euler_aapl_medium`: pipeline healthy; end-to-end runtime about `23.5m`.
  - `euler_full_12h`: first full run failed in PPO due to CUDA OOM; fixed by moving only minibatches to GPU.
  - `euler_full_tuned_24h`: stronger budget improved pretrain (`AAPL F1 0.687`, `GOOGL F1 0.610`) and made `GOOGL` slightly profitable, but still far from the paper.
  - Stage-2 AAPL sweep: balanced pretraining solved the class-collapse issue; `ctrl`/`sampler` reached about `0.64` val/test F1, but PPO still trailed `AS`.
  - Stage-3 AAPL sweep: better pretraining stayed stable, but PPO remained too passive; lowering `zeta` or freezing the backbone had little effect.
  - Stage-4 AAPL sweep: `pnl_inventory` reward with `60s` episodes and much higher `gamma`/`GAE` worked best; `euler_aapl_stage4_pnlinv60_ultra` reached `pnl 0.01469`, `nd_pnl 0.13978`, `sharpe 0.3014`, close to `AS`.
  - Stage-5 AAPL sweep: larger RL budget and extra reward shaping hurt; competitive quotes were the only promising change.
  - Stage-6 AAPL sweep: stage-4-sized budget plus competitive quotes was the best current result; `euler_aapl_stage6_ultra_competitive_ckpt` reached `pnl 0.01406`, `nd_pnl 0.14454`, `sharpe 0.2576`, while the plain control lagged clearly.
  - Stage-7 full runs: `AAPL` stayed mildly profitable and consistent across seeds, but `GOOGL` PPO had `0` fills and `0` PnL in both runs.
  - Stage-8 full runs: `AAPL` again landed at about `pnl 0.0106`, `nd_pnl 0.105-0.112`, `sharpe 0.244`, confirming the mainline result is real but not improving further; `GOOGL` remained weak, with one seed still at exact zero fills and the other only barely positive.
  - Corrected `lobmmx` stage-2 runs: multitask pretraining stayed healthy (`best_f1 = 0.645`), and `spread_aggr` reached `pnl 0.00954`, `nd_pnl 0.11283`, `sharpe 0.2579`, nearly matching mainline `AAPL` PPO while using higher bias and tighter spreads.
  - `lobmmx` still has a reward-scale issue: every creative variant, and even the creative baselines, retained strongly negative `reward`, so selection and optimization targets are still misaligned with profitability.
  - Storage/share result: the raw cluster `data/` tree is `84G`, but the compressed archive is only `12G`, which fits comfortably in the home quota and is now available under `/cluster/home/piroth/public/mlfcs-gapa/` for teammates who know the path.
  - Global conclusion so far: competitive quote scales matter a lot, deterministic evaluation was necessary, pretraining is now good enough, mainline `lobmm` is near its current AAPL ceiling, and the highest-upside work is now AAPL-focused `lobmmx` reward/selection calibration rather than more generic PPO tuning.
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

## Week of 2026-04-14

### Synthetic Continuous Branch

- Implemented a new isolated `lobmmsim/` package on branch `codex/simulated-continuous-paper`.
- Added a lightweight event-driven synthetic top-10 LOB generator that writes paper-like processed day folders plus `latent.csv` sidecar metadata.
- Added a paper-faithful continuous environment:
  - 2-action continuous quoting
  - `2000`-event episodes
  - zero initial cash/inventory
  - terminal liquidation
  - hybrid reward using dampened PnL, trading PnL, and inventory penalty
- Added synthetic-data pretraining, PPO training, baseline evaluation, and report generation on top of the existing PyTorch Attn-LOB / PPO core.
- Added tests for:
  - simulator determinism and book invariants
  - reward-component correctness and terminal liquidation
  - end-to-end synthetic smoke pipeline
- Verification:
  - `uv run pytest tests/test_lobmmsim_simulator.py tests/test_lobmmsim_env.py tests/test_lobmmsim_smoke.py -q`
  - result: `5 passed`
- Follow-up calibration work:
  - Retuned the synthetic event model so latent alpha acts more like a future-drift signal instead of immediate same-direction adverse selection.
  - Added a synthetic-specific PPO action prior so the continuous policy starts with neutral bias but a much narrower spread and actually receives fills.
  - Verified on a branch-level one-symbol smoke run for `000001` that the market is now economically sane:
    - `Fixed_1` profitable with `pnl_mean ~= 1544.5`
    - `OracleAlpha` profitable with `pnl_mean ~= 3578.5`
    - PPO still negative in smoke, so the simulator is now usable for research but not yet a solved learning problem.
