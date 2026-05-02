# Ghali's Work Journal — MLFCS RL Market-Making

## Setup

- **Local repo:** `/Users/ghali/MLFCS-Clone/mlfcs-gapa`, branch `mm-drl-lob`
- **Personal cluster:** `gberbich@student-cluster2.inf.ethz.ch`, working directory `/home/gberbich/simulation` (ETH 3DV class allocation — long jobs allowed)
- **Team cluster (Anja):** Euler HPC, `apetric@euler.ethz.ch`, results under `/cluster/scratch/apetric/artifacts_anja`
- **Background:** Limited finance knowledge, limited practical RL experience — learning as we go

---

## Day 1 — 2026-04-23/24: Understanding the Simulation

### What this project is

Train a reinforcement learning agent to do **market making** on a synthetic limit order book.
Market making = continuously posting buy and sell orders and profiting from the spread between them.
The risk = informed traders who know where the price is going will trade against you at a loss.

### The pipeline (4 stages, run in order)

1. `simulator.py` — generates fake market data
2. `pretrain.py` — trains a neural net (Attn-LOB) to predict short-term price direction
3. `train.py` — trains the RL agent (PPO) using the pretrained backbone
4. `report.py` — evaluates PPO against baselines on test data

### The simulator

Event-driven (not time-driven). At each step, one of 10 events is picked based on probabilities recomputed every event:

**3 types of actors:**
- **Noise traders** — trade for internal reasons (rebalancing etc.), no directional information
- **Informed traders** — have a signal, know which way price is going
- **Competing market makers** — post/cancel quotes, partially sense the signal

**Hidden state (agent never sees this):**
- `regime` ∈ {-1, 0, 1} — market trend direction, switches every ~300 events
- `signal` — drifts toward regime × alpha_signal_scale, with noise. Drives informed traders.
- `fair_value` — the true price. Moves with signal + noise + mean-reversion toward displayed mid.
- `vol_state` — GARCH-like volatility multiplier [0.6, 1.8]. Makes activity bursty, not constant.

**Key mechanics:**
- Informed buy probability: `informed_taker_rate × exp(0.55 × signal_edge) × hawkes_boost`
- `signal_edge` = (fair_value - midprice) / tick — how far true price is from displayed price
- `hawkes_boost` = 1 + hawkes_alpha × informed_clock — informed trades cluster (pack behavior)
- `lob_leak` — when signal is strong, book is refilled asymmetrically (more on signal side) — gives agent a hint in book shape before informed trades arrive
- Touch replenishment — noise trades refill the consumed price level instantly; informed trades do not (they cause real price discovery)

**Key config parameters:**
| Parameter | Default | What it controls |
|---|---|---|
| `informed_taker_rate` | ~0.35 | How often informed traders fire |
| `noise_taker_rate` | ~1.1 | How often noise traders fire |
| `market_order_alpha_impact` | 0.0004 | How much informed trades poison the book |
| `lob_leak_strength` | 0.3 | How visible the book imbalance is before informed trades |
| `informed_hawkes_alpha` | 0.1 | How much informed trades cluster |
| `informed_hawkes_decay` | 0.97 | How fast the clustering effect decays (~30 event half-life) |

### What the RL agent sees and does

**Observations:**
- Last 50 LOB snapshots (normalized prices + volumes) — input to Attn-LOB backbone
- 24 market features: Realized Volatility, RSI, OSI at multiple windows
- 24 agent-specific features: inventory, time remaining, spread, imbalance, cash

**Actions (continuous, Beta distribution):**
- `bias` — how much to shift quotes toward/away from mid based on inventory
- `spread` — how wide to quote

**Reward:**
- Dampened PnL (positive PnL from inventory moves is halved — discourages speculation)
- Trading PnL (reward for filling at good prices relative to mid)
- Inventory penalty (quadratic — punishes holding large positions)

### Baseline: Avellaneda-Stoikov (AS)

Classical mathematical formula for optimal market making. No ML. Used as:
1. The bar the RL agent needs to beat
2. A health check — if AS has negative PnL, the simulator is broken

### Results from Anja's Euler runs

| Config | PPO Sharpe | AS Sharpe | Notes |
|---|---|---|---|
| No leak, no hawkes | 0.35-0.47 | — | Agent learns almost nothing |
| leak=0.3, hawkes=0.1 | 1.25 | 4.04 | Big jump |
| leak=0.5, hawkes=0.1 | 1.19 | 3.74 | Slightly worse |
| + corpus normalization fix | 1.25 | — | test_f1 improved 0.11→0.14 |
| Full mode (running) | ? | ? | Job 64483855 on Euler |

**Sharpe ratio** = mean(PnL) / std(PnL). Higher = more consistent profits.
AS Sharpe still beats PPO Sharpe by ~3x. Closing this gap is the goal.

### Phase 1 fixes (committed, not yet tested at scale)

- Dropout added to Attn-LOB backbone (reduces overfitting)
- Label smoothing in pretraining
- Class-weighted loss (handles class imbalance)
- Corpus-wide volume normalization (main overfitting fix — already validated in medium mode)

### 3 gates to clear

1. `test_f1 ≥ 0.30` — pretrain model generalizes to unseen data
2. `lob_imbalance_future_return_corr[50] ≥ 0.10` — signal in book is learnable
3. `PPO Sharpe > AS Sharpe` — RL beats classical formula (main goal)

Current state: none cleared yet. test_f1 is at 0.14, PPO Sharpe at 1.25 vs AS 2.35-4.04.

---

## Day 2 — 2026-04-24: Planning Improvements

### Key concepts clarified

**F1 score** — between 0 and 1. Measures how well a classifier works, combining precision (when it predicts something, how often is it right) and recall (how often does it catch the true cases). 1.0 = perfect, 0.0 = useless. Here it measures how well the backbone predicts price direction (up/flat/down).

**Backbone (Attn-LOB)** — the neural net that reads the last 50 snapshots of the book and outputs a 64-number summary of what it sees. The RL agent builds on top of this summary to make decisions. Trained separately first (pretraining) on price direction prediction. If it overfits during pretraining, the RL agent gets garbage features — a broken foundation.

**Current backbone problem:** train F1 = 0.7, test F1 = 0.14. It memorized training data, not general patterns.

### Main bottleneck: backbone overfitting

The pretrain task was: predict midprice direction (up/flat/down) 10 events ahead. This signal is too noisy — 10 events is too short for any real price movement to appear. Result: test_f1 stuck at 0.14 regardless of regularization fixes.

### Change 1: Signal labels instead of price labels

Since we control the simulator, we have access to `latent_alpha` — the hidden signal that drives informed traders. Instead of asking the backbone to predict noisy price direction, we ask it to predict whether the signal is positive or negative (threshold = 0.1).

**Why this works:** The book shape (via LOB leak) is literally designed to hint at the signal. So this is a much more learnable task.

**Code change:** Added `_signal_labels()` in `data.py`. New config parameters: `pretrain_label_source=signal`, `pretrain_signal_threshold=0.1`.

**Smoke result:**
| | Price labels (baseline) | Signal labels (ours) |
|---|---|---|
| test_f1 | 0.14 | **0.257** |

Almost doubled. Close to the 0.30 target. Medium run submitted (job 54616) to confirm.

**Medium results (job 54616):**
| | Baseline | Signal labels |
|---|---|---|
| test_f1 | 0.142 | 0.199 |
| PPO Sharpe | 1.25 | 1.25 |

PPO Sharpe unchanged — medium mode is too small to show RL improvement. The validation metric was stuck at 4.0 every epoch across all runs, meaning the agent barely trains. Medium mode is only useful for measuring pretrain quality.

### Frozen backbone experiment

Hypothesis: if backbone is trainable during RL, PPO gradients may destroy the signal-detection features learned during pretraining. Freezing preserves them.

Result: identical to unfrozen — PPO Sharpe 1.25. Medium mode can't distinguish.

### Conclusion from medium experiments

Medium mode cannot show RL improvement regardless of changes. Only full mode has enough data and epochs for the agent to learn.

**Full run submitted:** job 54728 — signal labels + frozen backbone, full mode (21 days, 120k events/day). This is the real test.

**Full run results (job 54728):**

| | Anja's best (medium baseline) | Our full run |
|---|---|---|
| test_f1 | 0.14 | 0.19 |
| PPO Sharpe | 1.25 | 0.83 |
| AS PnL | 112 | 79.7 |
| bias_alpha_corr | -0.029 | -0.062 |
| selected_epoch | 0 | 0 |

**PPO Sharpe dropped from 1.25 to 0.83 — worse than baseline.**

Note: the 1.25 baseline is medium mode; full mode has more data, more episodes, and a harder environment. A direct numerical comparison may not be meaningful. This is the first and only clean full-scale result.

**What we observe:**
- `selected_epoch: 0` in every run means the best checkpoint is always the initial one — the agent does not improve during PPO training. This suggests the training is too short, the learning rate is too high, or the reward signal too sparse at full scale.
- `bias_alpha_corr: -0.062` means the agent is weakly skewing quotes in the wrong direction relative to the latent signal — it is not learning to use it.

**What we do not know:** whether the frozen backbone is helping or hurting. This run confounds backbone freeze with the move to full scale. There is no controlled unfrozen full-scale experiment with signal labels to compare against.

**Key lesson:** medium mode cannot evaluate RL improvements — it always returns selected_epoch=0 and identical metrics. Only full mode is meaningful for RL, but full mode takes 4-8h per run.

### New direction: calibrate simulator to match paper's stocks

The paper uses Shenzhen Stock Exchange data for 3 stocks: 000001 (Ping An Bank, ¥12.5), 000858 (Wuliangye, ¥135), 002415 (Hikvision, ¥32). Our simulator uses hand-tuned parameters that may not match the real microstructure of these stocks. Suspected cause of poor results vs paper: the synthetic environment doesn't match the real data the paper trained on.

---

## Simulator Calibration Study

### Motivation

The simulator parameters were hand-tuned without reference to real 000001 microstructure. If the synthetic market dynamics don't match the real stock, the agent learns strategies that exploit simulator artifacts rather than real patterns. We compared simulator output against real 000001 data and tuned parameters to close the gap.

### Real data

- **Stock:** 000001 (Ping An Bank, Shenzhen)
- **Source:** AKShare (accessible from ETH cluster only — blocked by VPN from Switzerland)
- **Period:** March–April 2026 (~32 trading days, 1184 bars). The paper uses November 2019 data but AKShare free tier only serves recent data; 000001 microstructure is stable over time.
- **Granularity:** 5-minute OHLCV, stable hours only (10:00–11:30 and 13:00–14:30, matching the paper)

### Microstructure metrics: before and after calibration

All simulator runs use medium mode (20k events/day × 8 days = 320 bars at 500-event window).

| Metric | Real 000001 | Before (run1) | After (run2) | Change |
|---|---|---|---|---|
| Return std per bar (%) | 0.195 | 0.198 | 0.240 | ➡ slight drift |
| Return kurtosis (fat tails) | 219 | 1.26 | 0.81 | ❌ still near-gaussian |
| Return autocorr lag-1 | -0.030 | -0.076 | +0.032 | ✅ much closer to real |
| \|Return\| autocorr (vol clustering) | 0.053 | 0.120 | 0.224 | ❌ worsened |
| Spread mean (ticks) | ~1 | 1.45 | 1.46 | ❌ unchanged |
| Regime switches/day | — | 146 | 21 | ✅ critical for RL |

### Parameter changes and reasoning

**`config.py`:**
- `informed_hawkes_alpha`: 0.1 → **0.04** — informed trades cluster less. Each informed trade self-excites the next; reducing this lowers the burst pattern that drives excess vol clustering.
- Added `regime_min_duration: int = 2000` — regime must persist at least 2000 events before it can switch. At 120k events/day and 2000-event episodes, the original 300-event minimum meant the regime changed 2–3 times per episode, making it impossible for the agent to learn to use it.
- Added `regime_switch_prob: float = 0.001` — per-event switch probability (was effectively 0.004), bringing switches/day from 146 to 21.
- `vol_target` baseline: `1.0 + 0.3*|regime|` → `0.7 + 0.3*|regime|` — allows vol_state to reach 0.7 in neutral regimes (genuinely calm periods).

**`simulator.py`:**
- Regime switch condition reads from config instead of hardcoded values.

### Calibration iterations

Six simulator configurations were tested (runs 1–6). Only the first change (run1→run2) produced measurable improvements. Runs 3–6 were all no-ops:

- **run3** (vol_target range): no effect. `0.99^500 ≈ 0` — vol_state fully converges to its target within every 500-event bar. Bar-level statistics are determined by the average regime state during that bar, not by intra-bar GARCH dynamics.
- **run4** (persistence 0.99→0.95): no effect for the same reason. `0.95^500 ≈ 0` too.
- **run5** (vol_target via signal instead of regime): no effect. Signal is a noisy proxy of regime — they are strongly correlated. Replacing regime with signal in the vol_target formula produces identical bar-level statistics.
- **run6** (random jumps + tighter spread): jumps at probability 0.00007/event = ~8/day total across 8 days (11 events) — too rare to affect 320-bar statistics. Spread narrowing: no measurable effect on spread metric.

The core insight: **any variable correlated with regime produces the same vol clustering** because long regimes (2000+ events) create sustained high-vol periods spanning multiple consecutive bars. Vol clustering and kurtosis cannot be improved without redesigning the regime mechanism itself, which would break the RL signal.

### Final calibration

**Adopted configuration (run2):** `regime_min_duration=2000`, `regime_switch_prob=0.001`, `hawkes_alpha=0.04`, `vol_target=0.7+0.3*|regime|`.

Two metrics meaningfully improved:
- **Return autocorr: -0.076 → +0.032** (real: -0.030) — price dynamics no longer over-correct after every move
- **Regime switches/day: 146 → 21** — agent now has a realistic chance of learning within a 2000-event episode

Remaining gaps (kurtosis, vol clustering, spread) are accepted as structural limitations.

---

### Effect on backbone (pretrain)

Calibration was tested in medium mode (signal labels, `pretrain_label_source=signal`):

| Config | test_f1 |
|---|---|
| Before calibration, medium, signal labels (job 54616) | 0.199 |
| After calibration, medium, signal labels (job 56471) | 0.199 |
| After calibration, full mode, price labels (job 56673) | 0.093 |
| After calibration, full mode, signal labels (job 57095) | 0.109 |

**Calibration had no measurable effect on backbone F1 in medium mode.** The 0.199 result comes from signal labels — the backbone is asked to predict the latent signal direction (which LOB leak makes observable in book shape), not price direction.

The full-mode runs with calibrated data both fail to learn:

- **Price labels (job 56673):** F1=0.093, expected — 67% flat class at 120k events/day makes the task unlearnable by construction.
- **Signal labels (job 57095):** F1=0.109 despite balanced classes [40% up / 20% flat / 40% down]. The model predicts flat for every input from epoch 1 and never improves across all 10 epochs.

The cause is a side effect of the calibration itself. With `regime_min_duration=2000` at full scale (120k events/day), each regime lasts ~5700 events on average. Within such a long regime, informed traders have enough time to push midprice to fair_value — price discovery completes, `signal_edge ≈ 0`, and LOB leak (which is proportional to signal_edge) drops to near zero. By the time most 50-event training windows are sampled, the LOB is close to balanced even though the latent signal is strongly directional. The backbone sees a balanced book and cannot determine signal direction. In medium mode (20k events/day, ~1000 events per regime), price discovery is incomplete for a larger fraction of events, so the imbalance remains visible and the backbone learns.

**The calibration that made regimes learnable for the RL agent made the LOB less readable for the backbone.**

---

### Effect on trained agent

| Run | Mode | Labels | Backbone | Pretrain F1 | PPO Sharpe | AS Sharpe | Trades/ep | Fill rate |
|---|---|---|---|---|---|---|---|---|
| job 54616 (pre-cal.) | medium | signal | unfrozen | 0.199 | 1.25 | 4.04 | 2.2 | 0.1% |
| job 56471 (post-cal.) | medium | signal | unfrozen | 0.199 | 1.25 | 4.16 | 2.2 | 0.1% |
| job 54728 (pre-cal.) | full | signal | frozen | 0.190 | 0.83 | — | ~2 | ~0.1% |
| job 56673 (post-cal.) | full | price | unfrozen | 0.093 | 0.45 | 2.04 | 0.55 | 0.027% |
| job 57095 (post-cal.) | full | signal | frozen | 0.109 | pending | — | — | — |

**Medium mode:** calibration had no effect on agent Sharpe (1.25 both times). Medium mode cannot show RL improvements — too few training episodes for PPO to converge.

**Full mode, pre-calibration (job 54728):** best full-scale result so far. Pretrain F1=0.19, Sharpe=0.83. The reference point for any post-calibration comparison.

**Full mode, post-calibration (jobs 56673 and 57095):** both fail at backbone learning. Job 56673 failed due to wrong label choice (price labels). Job 57095 fails despite correct signal labels — backbone F1=0.109, model predicts flat for every input from epoch 1 with no improvement across all 10 training epochs. Root cause: long-regime calibration makes the LOB unreadable (see backbone section above).

**What calibration did and didn't do:**
- ✅ Regime dynamics are more learnable for RL (21 vs 146 switches/day within 2000-event episodes)
- ✅ Price autocorrelation closer to real data (-0.030 real vs +0.032 sim, was -0.076)
- ❌ LOB readability broken at full scale — long regimes allow price discovery to complete, erasing the LOB imbalance the backbone needs to learn signal direction
- ❌ No improvement in agent Sharpe at any scale
- ❌ Clean post-calibration full-scale result still not achieved

**Next step:** run full mode on pre-calibration data with signal labels and unfrozen backbone — the one combination not yet tested. This isolates whether the backbone freeze or the calibration itself is the limiting factor in Sharpe 0.83, and gives a realistic ceiling before attempting further calibration fixes.
