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


### Contributor Update: Anja

- Focus area: Sharpe ratio computation in `lobmmx`.
- Context:
  - The paper (Guo et al., IJCNN 2023) computes Sharpe as `mean(episode_pnls) / std(episode_pnls)` across all test episodes. This is not annualized and not aggregated by day — the number depends on how many episodes you have and how long they are, so it is not comparable across setups or to standard finance Sharpe ratios.
  
- Completed:
  - Added two annualized Sharpe functions in `lobmmx/metrics.py` alongside the original `sharpe()` (kept for backward compatibility):
    - `sharpe_annualized_episodes(values, episodes_per_day)`: scales the per-episode Sharpe by `sqrt(episodes_per_day * 252)`. Useful as a diagnostic but not the recommended headline metric.
    - `sharpe_daily(pnls, days)`: groups episode PnLs by day, sums within each day, then computes `mean(daily_pnl) / std(daily_pnl) * sqrt(252)`. This is the industry-standard definition and the recommended primary metric.
  - Updated `summarize_results()` in `lobmmx/pipeline.py`:
    - now emits `sharpe_annual_ep` and `sharpe_annual_daily` alongside the original `sharpe` in every summary dict.
   - Updated `run_report()` in `lobmmx/report.py`:
    - `method_summary`, `symbol_summary`, and `paper_table` now compute and include `sharpe_annual_daily` alongside the original `sharpe`.
  - Updated `_format_overall_results()` in `lobmmx/report.py`:
    - computes and includes `sharpe_annual_daily` and `sharpe_annual_ep` columns in both the raw summary DataFrame and the formatted output table.
  - Same changes applied to `lobmm/metrics.py`, `lobmm/pipeline.py`, and `lobmm/report.py` for consistency.
  - All existing results remain comparable — the original `sharpe` field is unchanged, the new fields are additive.
- Links:
  - `lobmmx/metrics.py`, `lobmmx/pipeline.py`, `lobmmx/report.py`
  - `lobmm/metrics.py`, `lobmm/pipeline.py`, `lobmm/report.py`


## Week of 2026-04-04

### Weekly Snapshot

- Overall status: `lobmmx` fill model replaced with a queue-position-aware model that uses the L3 (MBO) cancellation data from Databento. The original model was too generous for US large-cap equities.
- Main goal for the week: Fix the fill simulator as it was the weakest link, make use of Level 3 (MBO) data and improve the model to make it aware of the position in the queue.
- Biggest win: Replaced the paper-style probabilistic fill rule with a queue-aware model that uses actual cancellation flow from `msg.csv`.
- Biggest risk or blocker: The more realistic `queue/back` setting produces much sparser fills, which may make PPO training harder. This likely requires either: training under `queue/uniform` and evaluating under `queue/back`, or increasing `trade_unit` and retuning inventory and reward scaling.

### Contributor Update: Anja

- Focus area: Making the `lobmmx` fill simulator more realistic for NASDAQ data.
- Context:
  - The paper (Guo et al., IJCNN 2023) was designed for the Shenzhen Stock Exchange, where tick sizes are sub-penny relative to the stock price, spreads are wider, and the microstructure is fundamentally different from US large-cap equities.
  - When we apply the same simulator to AAPL and GOOGL using Databento Level 3 (MBO) data, the fill model produces unrealistic results.
  - The original fill model uses a simple probabilistic rule: when a trade occurs at the agent's quoted price, the fill probability is `P(fill) = traded_volume / (traded_volume + depth)`. This formula implicitly assumes the agent's order is uniformly distributed in the queue. In reality, the agent's order was just placed and sits at the back of the queue.
  - This update proposes a new queue-position-aware model that uses the Level 3 (MBO) data. The key idea is: when the agent places an order, it joins the back of the queue at that price level. It only gets filled when the incoming trade volume exhausts all orders ahead of it.
- Findings from AAPL sample day analysis (20260302):
  - When 32 shares trade at ask1 against 91 shares of displayed depth, the original model gives `P(fill) = 32 / (32 + 91) = 26%`.
  - Under a back-of-queue model, the fill is `0%` because the trade does not reach through the 91 shares already ahead of the agent.
  - In AAPL, `0%` of trades were trade-throughs, so the original guaranteed-fill path (`if np.any(better)`) never triggers.
  - Conclusion: original model is too generous.
- Completed:
  - Implemented a new queue-position-aware fill model in `lobmmx/env.py`:
    - Replaced `_match_one_side` with a dispatcher between `_match_legacy` (original, preserved for backward compatibility) and `_match_queue` (new).
    - `_match_queue` processes each fill decision in three steps:
      1. Taker check: if the agent's price crosses the spread (e.g. ask quote ≤ best bid), it executes immediately as a taker. This is unchanged from the original.
      2. Trade-through check: if any trades at this event executed at a price worse than the agent's quote (from the aggressor's perspective), the agent's order is guaranteed filled. In practice this never happens for AAPL at the inside spread.
      3. Queue position check: for trades at the agent's exact price level, we compute how many shares were ahead of the agent in the queue. The agent only gets filled by the portion of traded volume that penetrates past the queue ahead.
    - Added `_queue_ahead` for queue position estimation:
      - When the agent places an order at `t_quote`, the displayed depth at that price is treated as queue ahead.
      - Between `t_quote` and the fill-check time `t_event`, cancellations reduce queue ahead.
      - Cancellation attrition is estimated from Level 3 message data in `msg.csv`: `attrition = total_withdrawals * (depth_at_our_level / total_side_depth)`.
      - This is approximate because withdrawals are aggregated across price levels (not per-level), but it uses real information available from the MBO feed (so it is directionally correct).
      - Effective queue ahead becomes: `queue_ahead = max(0, depth_at_placement - attrition)`.
    - Unlike the original model (all-or-nothing fill), the queue model naturally supports partial fills.
  - Added `msg` array (raw per-event message flow columns) to `DayData` in `lobmmx/data.py`:
    - Previously this data was loaded from `msg.csv` for computing rolling dynamic features (OSI, RV, RSI) and then discarded.
    - Now retained on `DayData` so the environment can access `withdraw_buy_volume` and `withdraw_sell_volume` for queue attrition.
    - Columns follow `MSG_COLUMNS` order: market buy/sell volume and count, limit buy/sell volume and count, withdraw buy/sell volume and count.
  - Added `fill_model` and `queue_position` configuration parameters in `lobmmx/config.py`:
    - `fill_model`: `"legacy"` (original paper model) or `"queue"` (new, default).
    - `queue_position`: `"back"` (agent joins end of queue, most realistic) or `"uniform"` (random queue position, more optimistic).
  - All previous runs used the legacy fill model and remain valid for comparison by setting `fill_model="legacy"`.
- Links:
  - `lobmmx/env.py`
  - `lobmmx/data.py`
  - `lobmmx/config.py`