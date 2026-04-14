# Automation State

Last updated: 2026-04-14 (Europe/Zurich)

## Canonical Paths

- Local repo: `/Users/piroth/Documents/projects/mlfcs-gapa`
- Euler repo: `/cluster/home/piroth/mlfcs-gapa`
- Canonical cluster data: `/cluster/work/math/piroth/mlfcs-gapa/data/processed`
- Canonical cluster artifacts: `/cluster/project/math/piroth/mlfcs-gapa/artifacts`
- Canonical cluster logs: `/cluster/project/math/piroth/mlfcs-gapa/logs`
- Canonical compressed data backup: `/cluster/work/math/piroth/mlfcs-gapa/data_20260330.tar.zst`
- Public teammate-facing archive copy: `/cluster/home/piroth/public/mlfcs-gapa/data_20260330.tar.zst`

Rules:
- Do not use the laptop dataset for real runs.
- Prefer the permanent `work` / `project` paths over scratch.
- Keep the Euler checkout clean so `git pull --ff-only` works.
- The live `work` data path is still private in practice because `/cluster/work/math/piroth` is not traversable by other users.
- Use the public home-archive copy for collaborator sharing; do not assume teammates can read the live `work` path.

## Current Best Known Setup

### Mainline `lobmm` best AAPL configuration

Best current result family: stage-6 competitive runs.

Key settings:
- `reward_mode = pnl_inventory`
- `target_episode_seconds = 60`
- `gamma = 0.99999`
- `gae_lambda = 0.9995`
- `ppo_lr = 3e-5`
- `ppo_epochs = 14`
- `ppo_rollouts_per_epoch = 128`
- `ppo_updates = 2`
- `ppo_minibatch_size = 2048`
- `max_train_episodes_per_day = 128`
- `max_eval_episodes_per_day = 16`
- `normalize_advantages = true`
- `gradient_clip_norm = 0.5`
- `backbone_trainable = true`
- `max_spread_bps = 7.0`
- `max_bias_bps = 3.0`
- `max_inventory = 250`
- `zeta = 0.004`
- `pretrain_balance_mode = balanced_sampler_and_loss`
- `pretrain_horizon = 10`

Best evidence:
- `euler_aapl_stage6_ultra_competitive_ckpt`
  - `pnl = 0.01406`
  - `nd_pnl = 0.14454`
  - `sharpe = 0.2576`
- `euler_aapl_stage6_ultra_competitive`
  - `pnl = 0.01250`
  - `nd_pnl = 0.13095`
  - `sharpe = 0.2685`

Interpretation:
- Competitive quoting helped materially.
- Stage-4-sized RL budget worked better than the heavier stage-5 budget.
- Plain `pnl_inventory` is still the most reliable reward family in `lobmm`.

### Creative fork baseline

Experimental package: `lobmmx/`

Main ideas already implemented:
- random initial inventory
- terminal inventory allowed
- reward based on trading edge, not mark-to-market holding gains
- reward normalized in spread/tick units
- US-timescale dynamic windows
- separate directional bias and inventory skew action components
- multitask pretraining (`mid`, `spread`, `flow`)
- maker/taker fees
- deterministic evaluation including deterministic initial inventory
- aggressive PPO checkpointing / validation selection

Status:
- First corrected `lobmmx` batch finished.
- `spread_aggr` is the only creative variant that nearly matches the mainline `AAPL` PPO result.
- Reward scaling / model-selection signal in `lobmmx` is still not coherent enough to trust as a new default.
- Selective branch integrations have now been validated.
- Annualized Sharpe reporting is worth keeping.
- The new queue-aware fill model should remain opt-in; it is not a better default yet.

## Active Runs

- Active Euler runs from commit `049caa5`:
  - `euler_lobmmx_stage4_2000_control`
    - train: `63416021`
    - evaluate: `63416029`
    - report: `63416033`
  - `euler_lobmmx_stage4_2000_excess_penalty`
    - train: `63416035`
    - evaluate: `63416037`
    - report: `63416039`
  - `euler_lobmmx_stage4_2000_excess_potential`
    - train: `63416042`
    - evaluate: `63416047`
    - report: `63416051`
  - purpose: compare a fixed-`2000`-event random-inventory control against initial-inventory-relative inventory control with and without potential-based reward for reducing excess inventory
- Most recently completed:
  - integrated branch-validation runs:
    - `euler_lobmmx_stage3_legacy_control`
      - train: `63388924`
      - evaluate: `63388926`
      - report: `63388928`
      - PPO result: `pnl = 0.01185`, `nd_pnl = 0.13772`, `sharpe = 0.526`, `reward = -0.293`
    - `euler_lobmmx_stage3_queue_back`
      - train: `63388930`
      - evaluate: `63388932`
      - report: `63388934`
      - PPO result: `pnl = -0.00511`, `nd_pnl = -0.05741`, `sharpe = -0.0668`, `reward = -0.933`
    - `euler_lobmmx_stage3_queue_uniform`
      - train: `63388936`
      - evaluate: `63388938`
      - report: `63388940`
      - PPO result: `pnl = 0.02353`, `nd_pnl = 0.28562`, `sharpe = 0.271`, `reward = -6.161`
      - interpretation: higher raw PnL, but achieved via much higher churn and much worse terminal-inventory penalties
  - mainline full runs:
    - `euler_full_stage8_competitive_seed29`
    - `euler_full_stage8_competitive_seed41`
  - creative AAPL runs:
    - `euler_lobmmx_aapl_stage2_spread_base`
    - `euler_lobmmx_aapl_stage2_ticks_base`
    - `euler_lobmmx_aapl_stage2_spread_aggr`
    - `euler_lobmmx_aapl_stage2_spread_halfcost`
- Current transfer state:
  - a local copy of `data_20260330.tar.zst` is being downloaded into the laptop `Downloads` folder for external sharing upload

## Important Findings

- Real-run downsampling was a major early mismatch and was removed.
- Making the encoder trainable by default was more faithful to the older original code path.
- The original China-market action scales were too small for US data; bps-based quote scales worked much better.
- `2000` events in the US data corresponded to only seconds, not the paper’s few-minute regime. Shorter wall-clock episodes around `60s` worked best so far.
- Deterministic evaluation was necessary; before that, cross-run comparisons were noisy and misleading.
- Pretraining used to collapse to one class; balanced pretraining fixed this and brought val/test F1 to about `0.64+` on AAPL.
- Once pretraining stabilized, the main bottleneck shifted from encoder quality to RL policy behavior.
- PPO repeatedly learned overly passive, nearly symmetric quotes until competitive quote scales were introduced.
- Heavy RL budget increases regressed performance; more training was not automatically better.
- Checkpoint selection is still not trustworthy in mainline `lobmm`; checkpointed runs often selected `epoch 0`, so selection logic should not be treated as validated there.
- `cluster/submit_lobmmx_aapl.sh` had an env-ordering bug that let `lobmmx` evaluate/report inherit the pretrain dependency instead of the train dependency; fixed in commit `324770a`, and corrected downstream jobs were resubmitted as `61636126/28/30/32/34/35`.
- Stage-7 full runs finished cleanly and were highly consistent across seeds:
  - `AAPL` stayed mildly profitable with the stage-6 competitive setup.
  - `GOOGL` PPO produced effectively zero fills and zero PnL in both seeds.
- Stage-8 full runs finished cleanly and mostly confirmed stage-7:
  - `AAPL` remained reproducible across seeds at about `pnl = 0.0106`, `nd_pnl = 0.105-0.112`, `sharpe = 0.244`
  - `AAPL` PPO still trails `AS` and `Fixed_3`
  - `GOOGL` remained a weak transfer case:
    - seed 29: exact `0` fills and `0` PnL
    - seed 41: tiny positive result (`pnl = 0.00437`) but still almost no trades
- First `lobmmx` creative runs finished cleanly but underperformed mainline `lobmm`:
  - pretraining stayed healthy (`best_f1 ≈ 0.648`)
  - all three variants were slower than mainline and landed near `AAPL pnl ≈ 0.0055-0.0058`
  - the spread/tick reward variants produced strongly negative `reward_mean`, so the current inventory-scaled reward is dominating episode learning in an unhelpful way.
- `lobmmx` reward/selection fixes now in place:
  - default PPO selection metric changed from `reward_mean` to `pnl_mean`
  - default per-step inventory penalty removed for `trade_inventory`
  - terminal inventory is now penalized by explicit liquidation cost in spread/tick units
  - `PnLMAP`-related position stats now use inventory changes relative to the random initial inventory, not the raw starting position
  - local `lobmmx` smoke pretrain/train/evaluate/report path was exercised successfully against a temporary sample-backed dataset
- Selective branch integration outcome:
  - kept:
    - merged `lobmmx` reward-fix PR from mainline
    - annualized Sharpe reporting in `lobmm` and `lobmmx`
    - optional queue-aware `lobmmx` fill model behind `fill_model`
    - `diag_microstructure.py`
  - rejected as a new default:
    - queue-aware fill model
  - reason:
    - `queue_back` was too pessimistic and made PPO unprofitable
    - `queue_uniform` improved raw PnL but increased turnover, position size, and terminal penalties enough that the objective became much worse
    - current conclusion is to keep `fill_model = legacy` as the default and only revisit queue-aware fills together with reward/terminal-penalty retuning
- Corrected `lobmmx` stage-2 AAPL runs finished cleanly:
  - shared pretrain remained healthy (`best_f1 = 0.6450`)
  - `spread_aggr` was clearly best:
    - `pnl = 0.00954`
    - `nd_pnl = 0.11283`
    - `sharpe = 0.2579`
    - `avg_spread_bps = 3.34`
    - `avg_bias_bps = 0.33`
    - `fill_rate = 3.34e-05`
  - `spread_aggr` nearly matched mainline `AAPL` PPO while learning visibly more directional bias
  - but all `lobmmx` reward means stayed strongly negative, including baselines, so reward scale / selection calibration is still unresolved
- Data handoff path now in place:
  - raw cluster `data/` remains the live source
  - the old uncompressed backup was removed
  - compressed backup/share archive is `12G`
  - public copy under `/cluster/home/piroth/public/mlfcs-gapa/` fits safely within home quota and is the intended collaborator handoff path

## Experiment History And Results

Use this section to avoid rerunning dead ends. The point is not to perfectly log every metric, but to remember what each stage established.

### Early pipeline / systems fixes

- Real-run downsampling:
  - Initially, the effective “full” setup still sampled only a small subset of rows and episodes.
  - This was removed because it was a major mismatch to the intended event-by-event setting.

- Encoder training and smoke mode:
  - Encoder was changed to be trainable by default.
  - Smoke was aligned more closely to real mode so it differed mainly in budget, not in modeling choices.

- PPO memory bug:
  - First serious full run failed with CUDA OOM because PPO tried to move the whole rollout batch to GPU.
  - Fixed by moving only minibatches to device.

- Cluster storage:
  - Data, artifacts, and logs were moved from scratch to permanent `work` / `project` storage.

### `euler_aapl_medium`

- Purpose:
  - First bounded serious end-to-end check after the infrastructure stabilized.
- Outcome:
  - Healthy pipeline, end-to-end runtime about `23.5m`.
- Main lesson:
  - The stack was operationally sound enough to start more serious tuning.

### `euler_full_12h`

- Purpose:
  - First larger full run on both symbols.
- Outcome:
  - Failed in PPO due to CUDA OOM.
- Main lesson:
  - The PPO update path needed minibatch-only GPU transfer.
- Do not repeat:
  - Old rollout-to-GPU behavior.

### `euler_full_tuned_24h`

- Purpose:
  - Larger-budget full run with more RL, more data, and broader quote scales.
- Outcome:
  - Improved pretraining:
    - `AAPL F1 = 0.687`
    - `GOOGL F1 = 0.610`
  - PPO improved relative to earlier full runs.
  - `GOOGL` became slightly profitable, but overall still far from paper-level results.
- Main lesson:
  - More data and a somewhat wider US-appropriate action space helped, but did not solve the main RL problem.

### Stage-7 full two-symbol competitive runs

- Runs:
  - `euler_full_stage7_competitive_seed7`
  - `euler_full_stage7_competitive_seed13`
- Outcome:
  - very consistent across seeds
  - `AAPL PPO` remained mildly profitable:
    - seed 7: `pnl = 0.01063`, `nd_pnl = 0.11158`, `sharpe = 0.2436`
    - seed 13: `pnl = 0.00906`, `nd_pnl = 0.09143`, `sharpe = 0.2573`
  - `GOOGL PPO` failed behaviorally:
    - both seeds: `pnl = 0`, `fill_rate = 0`, `trades = 0`
- Main lesson:
  - the stage-6 competitive setup is stable on `AAPL`
  - it does not transfer to `GOOGL` without symbol-specific changes, most likely tighter quoting or spread-relative parameterization
  - another full AAPL+GOOGL run should not be launched before fixing `GOOGL`

### First `lobmmx` creative AAPL batch

- Runs:
  - `euler_lobmmx_aapl_spread_base`
  - `euler_lobmmx_aapl_ticks_base`
  - `euler_lobmmx_aapl_spread_alpha`
- Outcome:
  - shared multitask pretrain remained healthy: `best_f1 = 0.6475`
  - PPO was slower than mainline and weaker:
    - `spread_base`: `pnl = 0.00579`, `nd_pnl = 0.05992`, `sharpe = 0.1409`
    - `ticks_base`: `pnl = 0.00554`, `nd_pnl = 0.05675`, `sharpe = 0.1348`
    - `spread_alpha`: `pnl = 0.00579`, `nd_pnl = 0.05893`, `sharpe = 0.1409`
- Main lesson:
  - the large creative changes did not help yet
  - the current `lobmmx` reward construction is mis-scaled: profitable episodes still have strongly negative `reward_mean`
  - validation-based PPO selection is still weak in `lobmmx` because the selection metric is tied to that distorted reward

### Stage-2 AAPL sweep

- Purpose:
  - Fix pretraining collapse and test more serious AAPL-only tuning runs.
- Variants:
  - `base`
  - `ctrl`
  - `h20`
  - `sampler`
- Outcome:
  - Balanced pretraining solved the class-collapse issue.
  - Best val/test pretrain F1 moved to about `0.64`.
  - PPO still underperformed `AS`.
- Main lesson:
  - Pretraining was no longer the dominant blocker; policy learning became the bottleneck.
- Keep:
  - balanced pretraining
- Do not repeat:
  - unbalanced pretraining as the default

### Stage-3 AAPL sweep

- Purpose:
  - Test whether lower inventory penalty or frozen backbone would fix passive PPO behavior.
- Variants:
  - `sampler_base`
  - `sampler_lozeta`
  - `sampler_frozen`
  - `sampler_seed13`
- Outcome:
  - Pretraining remained strong and stable.
  - PPO still learned very low-fill, weak-bias policies.
  - Lower `zeta` and freezing the backbone had little effect on final quality.
- Main lesson:
  - The issue was not mainly the inventory penalty coefficient or whether the backbone was frozen.
- Do not prioritize:
  - more `zeta` micro-tuning
  - frozen-backbone as the main path

### Stage-4 AAPL sweep

- Purpose:
  - Test the hypothesis that much longer episodes caused RL issues, and adjust long-horizon PPO settings accordingly.
- Key changes:
  - `60s` episodes
  - much higher `gamma`
  - much higher `gae_lambda`
  - cleaner `pnl_inventory` reward
  - deterministic evaluation
- Variants:
  - `pnlinv60_main`
  - `pnlinv60_ultra`
  - `pnlinv120_main`
  - `hybridsafe60`
- Best result:
  - `euler_aapl_stage4_pnlinv60_ultra`
  - `pnl = 0.01469`
  - `nd_pnl = 0.13978`
  - `sharpe = 0.3014`
- Main lessons:
  - `60s` episodes were clearly better than `120s`.
  - high discount / GAE helped.
  - clean `pnl_inventory` beat hybrid reward variants.
- Keep:
  - deterministic evaluation
  - short wall-clock episodes
  - long-horizon PPO settings

### Stage-5 AAPL sweep

- Purpose:
  - Push RL budget higher and try additional reward-shaping ideas.
- Variants:
  - `ultra_plus`
  - `risk_ramp`
  - `l1l2_inventory`
  - `competitive_quotes`
  - `low_lr_long`
- Outcome:
  - Larger RL budget hurt.
  - Extra reward shaping hurt or did not help.
  - `competitive_quotes` was the only promising direction, and even it was only relatively better inside this batch.
- Main lessons:
  - more PPO budget is not automatically beneficial
  - extra reward shaping in mainline `lobmm` is low-value
- Do not repeat:
  - stage-5-style heavier PPO as the default
  - L1/L2 or ramped inventory penalties as the mainline default

### Stage-6 AAPL sweep

- Purpose:
  - Return to the successful stage-4-sized RL budget and test competitive quote scales with and without checkpoint selection.
- Variants:
  - `ultra_competitive`
  - `ultra_competitive_ckpt`
  - `ultra_control_ckpt`
- Best results:
  - `euler_aapl_stage6_ultra_competitive_ckpt`
    - `pnl = 0.01406`
    - `nd_pnl = 0.14454`
    - `sharpe = 0.2576`
  - `euler_aapl_stage6_ultra_competitive`
    - `pnl = 0.01250`
    - `nd_pnl = 0.13095`
    - `sharpe = 0.2685`
- Main lessons:
  - competitive quotes clearly helped
  - stage-4-sized budget was better than stage-5 heavy budget
  - checkpoint selection is still not validated because checkpointed runs often selected `epoch 0`
- Keep:
  - competitive quote scales
- Do not assume:
  - checkpoint selection is solved in `lobmm`

### Stage-7 full runs

- Purpose:
  - Exploit the current best mainline setup on both symbols with two seeds.
- Runs:
  - `euler_full_stage7_competitive_seed7`
  - `euler_full_stage7_competitive_seed13`
- Status:
  - Running at last update.
- Interpretation target:
  - confirm whether the stage-6 competitive setup generalizes to both `AAPL` and `GOOGL`
  - measure seed sensitivity under deterministic evaluation

### `lobmmx` creative fork

- Purpose:
  - Explore bigger structural changes that are awkward to keep layering into `lobmm`.
- Implemented ideas:
  - random initial inventory
  - allow terminal inventory
  - reward only trading edge, not buy-and-hold mark-to-market gains
  - spread/tick-unit reward scaling
  - US-timescale state windows
  - decoupled directional alpha and inventory skew
  - multitask pretraining closer to market making
  - maker/taker fees
  - deterministic evaluation including deterministic initial inventory
  - aggressive PPO checkpointing / validation selection
- First batch:
  - `euler_lobmmx_aapl_spread_base`
  - `euler_lobmmx_aapl_ticks_base`
  - `euler_lobmmx_aapl_spread_alpha`
- Status:
  - Running at last update.

## What Has Already Been Tried

This section is for fast duplicate avoidance.

- Data handling:
  - local large data copies for real runs: no, stop doing this
  - scratch as the canonical storage path: no, replaced by permanent storage

- Evaluation:
  - non-deterministic evaluation: already tried, caused misleading comparisons
  - deterministic evaluation: keep

- Pretraining:
  - unbalanced pretraining: bad, prone to class collapse
  - balanced pretraining: good, keep
  - horizon increase to `20`: tried, somewhat okay but not clearly better than the balanced `10`-horizon path

- Backbone policy:
  - frozen backbone in RL: tried, not the main lever
  - trainable backbone: preferred current default

- Episode design:
  - much longer episodes (`120s`): worse
  - about `60s`: best so far in mainline

- Reward ideas in mainline:
  - `pnl_inventory`: best current family
  - hybrid / safe hybrid: worse than clean `pnl_inventory`
  - ramped inventory penalty: tried, not beneficial
  - mixed L1/L2 inventory penalty: tried, not beneficial

- RL budget:
  - heavier stage-5 budget: worse
  - stage-4-sized budget: better

- Quote scales:
  - conservative / wider quotes: too passive
  - competitive quotes: good, keep exploring

- Checkpoint selection:
  - current mainline version: not trustworthy yet
  - creative fork version: implemented, but not yet validated

## Previous Stage Summary

- Before stage-2:
  - infrastructure and preprocessing were the main concerns
- Stage-2:
  - fixed pretraining collapse
- Stage-3:
  - showed that inventory penalty and freezing were not the main problems
- Stage-4:
  - identified the best current RL regime (`60s`, long-horizon PPO, `pnl_inventory`)
- Stage-5:
  - ruled out “just train harder” and “shape reward more”
- Stage-6:
  - identified competitive quoting as the strongest current-code improvement
- Stage-7:
  - tests whether the best AAPL setup holds at larger scale
- `lobmmx`:
  - starts the next major exploration branch with more structural changes

## Known Bad Ideas

- Stage-5 heavier RL budget:
  - More epochs, more rollouts, and more reward shaping made performance worse.
  - Do not repeat as the default path.

- Relying on non-deterministic evaluation:
  - This polluted run comparisons.
  - Do not compare new results to old non-deterministic runs without caution.

- Assuming better pretraining alone will solve PPO:
  - Pretraining is now “good enough” for the current path.
  - Remaining gains likely require environment/state improvements.

- Pure reward-shaping sweeps in `lobmm`:
  - Most did not help.
  - Do not prioritize more reward heuristics in the mainline path over environment realism.

## Open Hypotheses

- Competitive quoting is a real lever across symbols, not just AAPL.
- The mainline `lobmm` path may be near its ceiling without better environment realism.
- `lobmmx` may improve results because it removes mark-to-market holding gains, introduces inventory randomness, and uses MM-aligned normalization/tasks.
- Spread-unit vs tick-unit reward scaling may materially change PPO behavior in the creative fork.
- Larger directional alpha freedom may help because current PPO still learns very small bias magnitudes.

## Decision Rules

- If jobs are still running normally, do not interrupt them.
- If a job fails, inspect logs and artifact summaries before resubmitting anything.
- For `lobmm`, prefer the stage-6 competitive configuration as the default control.
- Do not treat `ppo_select_best_model` as validated in `lobmm` yet.
- If a new run improves only because evaluation settings changed, do not count it as a real improvement.
- Prefer changes to environment realism, state timescale, and quote parameterization over simply adding more PPO budget.
- Use `progress_report.md` for human-readable weekly summaries.
- Update this file when:
  - a new best setup is found
  - a run family is ruled out
  - canonical paths or operating rules change

## Next Recommended Actions

1. Let the active stage-7 and first `lobmmx` runs finish.
2. Compare:
   - stage-7 full `AAPL+GOOGL` current-code results
   - `lobmmx` spread-unit vs tick-unit vs more-directional creative variants
3. If `lobmmx` shows promise, keep iterating there in AAPL-only mode before scaling it to both symbols.
4. If stage-7 current-code runs are strong and stable, use that setup as the exploitation baseline while `lobmmx` remains the exploration branch.

## Useful Commands

- `ssh euler 'squeue --me'`
- `ssh euler 'sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode'`
- `ssh euler 'cd /cluster/home/piroth/mlfcs-gapa && git pull --ff-only'`
- `ssh euler 'cd /cluster/home/piroth/mlfcs-gapa && cluster/submit_euler.sh ...'`
- `ssh euler 'cd /cluster/home/piroth/mlfcs-gapa && cluster/submit_lobmmx_aapl.sh'`

## 2026-04-14 Synthetic Branch

- New active code path:
  - `lobmmsim/`
  - isolated synthetic-data branch intended to stay close to the paper's continuous implementation
- Implemented:
  - synthetic top-10 LOB generator with paper-style session windows and stable windows
  - per-event `latent.csv` sidecar with efficient price, latent alpha, regime, signed flow, queue pressure, and event labels
  - paper-style continuous environment with 2-action quote control and hybrid reward
  - synthetic pretraining, PPO training, and diagnostics/report pipeline
  - oracle latent baseline and fixed-level baseline for synthetic evaluation
- Verified locally:
  - `uv run pytest tests/test_lobmmsim_simulator.py tests/test_lobmmsim_env.py tests/test_lobmmsim_smoke.py -q`
  - result: `5 passed`
- Current branch:
  - `codex/simulated-continuous-paper`
- Current conclusion:
  - the synthetic path is runnable end to end and isolated from the main real-data experimentation code
  - next work on this branch should focus on calibration quality and whether learned policies recover the planted latent signal
