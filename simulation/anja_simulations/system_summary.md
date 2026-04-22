# Complete Technical Summary: RL Market-Making from LOB Data

This document explains the entire codebase in detail — how the simulator generates data, how the environment presents it to the RL agent, how training works, and what specific improvements were made and why. Use this as ground truth when making further code changes.

---

## 1. Project Overview

This project trains an RL agent (PPO with continuous actions) to do market making on a synthetic limit order book (LOB). The pipeline has four stages that run in order:

1. **`simulator.py`** generates synthetic LOB data by simulating an agent-based market
2. **`pretrain.py`** pretrains the Attn-LOB backbone on mid-price direction prediction (up/down/flat classification)
3. **`train.py`** trains the PPO agent using the pretrained backbone, evaluated against validation data
4. **`report.py`** evaluates the trained PPO agent against baselines (AS, Fixed_1, Fixed_2) on test data

The orchestration file `run_suite.py` runs all four in sequence. The file `sweep.py` runs multiple configurations to find the best hyperparameters.

The paper this implements is "Market Making with Deep Reinforcement Learning from Limit Order Books" (Guo, Lin, Huang — IEEE). The data format mirrors the Shenzhen Stock Exchange with three Chinese stocks: 000001 (Ping An Bank, ¥12.5), 000858 (Wuliangye, ¥135), 002415 (Hikvision, ¥32). All prices use tick_size=0.01 and trade_unit=100 shares.

---

## 2. The Simulator (`simulator.py`)

### 2.1 What it produces

For each symbol and each day, it generates CSV files:
- `ask.csv` / `bid.csv`: top-10 LOB snapshots (price + volume per level, per event)
- `price.csv`: ask1_price, bid1_price, midprice per event
- `msg.csv`: per-event message counts (market_buy_volume, limit_sell_n, etc.)
- `trades.csv`: every executed trade with taker_agent, maker_agent, queue_ahead
- `latent.csv`: hidden state (fair_value, signal, regime, vol_state, spread_ticks, etc.)

Each "day" has `events_per_day` events (default 120k for 000001, 90k for 000858, 60k for 002415). In `smoke` mode these are capped to 500, in `medium` mode to 20k.

### 2.2 Core architecture: AgentBasedLOB

The class `AgentBasedLOB` maintains an explicit order book with `bids` and `asks` as `dict[float, deque[RestingOrder]]`. Each `RestingOrder` has an `order_id`, `owner` (which agent type placed it), `side`, `price`, `size`, and `created_event`.

On each step, the simulator:
1. Calls `_step_latent()` to evolve the hidden state
2. Calls `_choose_event()` to probabilistically select one of 10 event types
3. Executes that event (market order, limit add, cancel, or refill)
4. Calls `_ensure_depth()` to maintain minimum book depth and narrow wide spreads
5. Takes a snapshot of the resulting book state

### 2.3 Hidden state dynamics (`_step_latent`)

Three coupled latent processes drive the market:

**Regime** (`self.regime` ∈ {-1, 0, 1}): A discrete state that switches rarely (after 300+ events, with 0.4% probability per step). Regime -1 = downtrend, 0 = neutral, 1 = uptrend. Distribution: 30/40/30.

**Signal** (`self.signal`): A persistent directional state that mean-reverts toward `0.4 * regime * alpha_signal_scale`. Updated as:
```
signal = persistence * signal + (1-persistence) * target + N(0, signal_noise * alpha_signal_scale)
```
Persistence is per-symbol (0.985 for 000001, 0.988 for 000858, 0.99 for 002415). The signal is what makes informed traders directional — it represents private information about fair value.

**Fair value** (`self.fair_value`): The latent "true" price, not directly observed by the RL agent. Updated each step as:
```
reversion = strength * (displayed_mid - fair_value)   # pulls FV toward book mid
drift     = price_scale * 0.0022 * signal              # signal-driven trend
noise     = N(0, price_scale * vol_state * 0.35 * price_noise_scale)
fair_value += reversion + drift + noise
```

The reversion is **nonlinear**: `strength = 0.003 + 0.04 * min(edge_ticks/3, 1)^2`. At 1-tick deviation it's nearly invisible (0.0034), at 3+ ticks it's strong (0.043). This prevents the fair value from diverging far from the displayed book (which would create a feedback loop where MMs retreat → spread widens → more divergence).

**`price_scale`** = `reference_price / 12.5` ensures that a ¥135 stock has proportionally larger price moves than a ¥12.5 stock, so all symbols produce similar percentage daily ranges.

**Volatility clustering** (`self.vol_state`): A GARCH-like multiplier in [0.6, 1.8] that scales the fair_value noise. It rises during trending regimes and mean-reverts toward 1.0 in neutral markets:
```
vol_target = 1.0 + 0.3*|regime| + 0.1*min(|signal|, 2)
vol_state = 0.99 * vol_state + 0.01 * vol_target + 0.003*|N(0,1)|
```

### 2.4 Event selection (`_event_weights`)

Each step, one of 10 events is chosen with probabilities that depend on the current market state:

| Event | Base rate | Modulation |
|---|---|---|
| `noise_market_buy/sell` | `noise_taker_rate` (~1.0-1.3) | Weakly depends on imbalance, flow, vol |
| `informed_market_buy` | `informed_taker_rate` (~0.28-0.45) | `exp(0.80 * signal_edge)` — buys more when FV > mid |
| `informed_market_sell` | `informed_taker_rate` | `exp(-0.80 * signal_edge)` — sells more when FV < mid |
| `maker_add_bid/ask` | `maker_add_rate` (~0.9-1.15) | Slight retreat during strong signal |
| `maker_cancel_bid/ask` | `maker_cancel_rate` (~0.75-0.9) | Stronger cancellation during strong signal (`exp(0.22*edge)`) |
| `refill_bid/ask` | `liquidity_refill_rate` (~0.85-1.1) | Very slight vol sensitivity |

`signal_edge = clip((fair_value - mid) / tick, -2.0, 2.0)`. The clip at ±2 prevents runaway rate amplification.

### 2.5 Market order execution (`_market_order`)

When a market order fires:
1. **Capture pre-trade touch price**: `pre_touch = self.best_ask` (for a buy). This is critical — see section 2.7.
2. **Size determination**: Noise takers draw from the standard lot distribution. Informed takers get a right-skewed multiplier: `1.1 + 0.3 * Exp(1)`, so they occasionally send larger orders.
3. **Walk the book**: FIFO matching, consuming from the best price level first, oldest orders first. Each match creates a `TradeRecord` with full provenance (taker_agent, maker_agent, queue_ahead).
4. **Update signed flow state**: Exponentially weighted running sum of signed volume. Decay 0.985 per event.
5. **Move fair value**: The impact is size-proportional:
   ```
   size_lots = trade_size / trade_unit
   informed_scale = 1.5 if informed else 0.5
   impact = market_order_impact_scale * price_scale * (
       tick_impact * tick * (0.7 + 0.3*size_lots)
       + informed_scale * alpha_impact * max(|signal|, 0.2)
   )
   ```
6. **Touch replenishment** (see section 2.7)

### 2.6 Depth management (`_ensure_depth`)

Called after every event. Does two things:

**Level filling**: For each side, ensures at least 8 contiguous price levels exist from the current best price outward. Missing levels get a single `liquidity_provider` order.

**Spread narrowing**: If spread > 1 tick, with probability `min(0.92, 0.55*(spread-1))`, competing MMs add limit orders at improved prices between the current best bid and best ask. This is the mechanism that keeps spreads realistic — without it, spreads can only widen (via consumption or cancellation), never narrow.

### 2.7 Touch replenishment — the critical fix

**The problem**: When a market order fully consumes the best ask level, `self.best_ask` jumps up by 1+ ticks. This immediately moves the midprice by 0.5 ticks. This happens even for noise trades that carry no information. The result is that every trade creates ~0.1 ticks of "phantom adverse selection" against the maker, making passive market-making structurally unprofitable.

**The fix**: Before the book walk, we capture `pre_touch = self.best_ask`. After the walk, if (a) `touch_replenish_fraction > 0`, (b) the taker was `noise_taker`, and (c) `self.best_ask` has moved up from `pre_touch`, we add a liquidity_provider order at the **original** `pre_touch` price. This simulates the real-market behavior where LPs instantly refill a consumed price level because the wide spread is a profit opportunity.

Importantly, we do NOT replenish after informed trades. Informed flow is supposed to move prices — that's real price discovery. Only noise flow gets replenished, which is economically correct: noise trades should be approximately zero-information.

**Measured result**: Noise-buy adverse drift went from +0.094 ticks to +0.009 ticks (effectively zero). Informed-buy adverse drift is preserved at +0.196 ticks.

### 2.8 Competing MM placement (`_place_competing_mm`)

When a `maker_add_bid/ask` event fires:
1. **Touch-join probability**: Base is per-symbol (0.68-0.78), reduced by a penalty proportional to signal edge and vol_state. Clamped to minimum 0.30.
2. **Depth ceiling**: If the touch already has depth > `8 * trade_unit * depth_scale`, the MM is forced to place deeper (no benefit to joining a thick queue).
3. **Spread-tightening**: If spread > 1 tick and the MM was going to join the touch, with probability `min(0.7, 0.2 + 0.15*(spread-1))` it instead *improves* the best price by 1 tick.
4. **Standard placement**: If not joining touch, places 1-3 levels deep with probabilities [0.55, 0.30, 0.15].

### 2.9 Cancellation targeting (`_cancel_from_side`)

Cancellations use a blended weight: 40% exponential decay from the touch (prefers cancelling near-touch orders), 60% proportional to level depth (prefers cancelling from thick levels). This prevents touch depth from growing unboundedly — MMs pull from queues where they have low priority (far back in a thick queue).

### 2.10 Symbol profiles

Each symbol gets a `SymbolProfile` with per-stock rates, scaled by config multipliers:
- 000001 (¥12.5): highest event rates, most liquid (`depth_scale=1.35`, `noise_taker_rate=1.3`)
- 000858 (¥135): mid-liquidity (`depth_scale=1.0`, baseline rates)
- 002415 (¥32): least liquid (`depth_scale=0.82`, lowest rates)

---

## 3. Data Loading (`data.py`)

`load_day()` reads the CSVs and computes:
- **`lob`**: Raw 40-column array (ask1_price, ask1_volume, bid1_price, bid1_volume, × 10 levels)
- **`normalized_lob`**: Prices as relative to mid (ask_p/mid - 1), volumes max-normalized per level. Shape: `(T, 40)`. This is what the CNN backbone sees.
- **`dynamic`**: 24 features computed by `features.py` — Realized Volatility (3 windows), RSI (3 windows), OSI in volumes and counts (6 features × 3 windows = 18). Total = 24.
- **`trades_by_index`**: Dict mapping event index → `TradeSlice` for the fill simulator in env.py.
- **`labels`**: Three-class mid-price direction labels for pretraining (up=0, flat=1, down=2), using smoothed future/past means with threshold `alpha=1e-5`.

`_paper_normalize_lob` normalizes prices as `ask_p / mid - 1` (making them relative and stationary) and volumes as `vol / max_vol_per_level` (max-norm across the day).

---

## 4. The Environment (`env.py`)

### 4.1 Episode structure

`ContinuousMarketEnv` wraps one day of data. It creates `decision_indices` starting from `lookback-1+latency` (so the agent always has a full lookback window). Episodes are non-overlapping chunks of `episode_length` events (default 2000, ~3-5 minutes of real market time).

### 4.2 Observation space

Each step, the agent observes:
- **`lob`**: shape `(50, 40)` — the last 50 normalized LOB snapshots. This is the input to Attn-LOB.
- **`flat`**: shape `(48,)` — concatenation of:
  - `dynamic[idx]`: 24 market features (RV, RSI, OSI at multiple windows)
  - `_agent_state(idx)`: 24 agent-specific features:
    - `inv_scaled` × 4: inventory / max_inventory, repeated
    - `remaining` × 4: fraction of episode remaining
    - `spread_norm` × 3: current spread in ticks / 10, clipped to [0,1]
    - `imbalance` × 3: (bid1_vol - ask1_vol) / (bid1_vol + ask1_vol)
    - `vol_norm` × 3: latent vol_state / 3, clipped to [0,1]
    - `cash_norm` × 3: cash / turnover, clipped to [-1,1]
    - `|inv_scaled|` × 2: absolute inventory level
    - `sign(inv) * spread_norm` × 2: directional spread interaction

The repeated features (×4, ×3, ×2) are a design choice that gives the network multiple input neurons per concept — the total is exactly 24 to match the dynamic feature count, keeping the flat input at 48 dimensions.

### 4.3 Action space

Two continuous actions in [0, 1], sampled from a Beta distribution:

- **`action[0]`** → **bias** (δ): `delta = action[0] * max_bias`. The reservation price is `p_r = mid - sign(inventory) * delta`. When inventory is positive, the reservation shifts down (encouraging sells); when negative, it shifts up (encouraging buys). `max_bias` default = 0.05.
- **`action[1]`** → **spread**: `spread = action[1] * max_spread`, clipped to ≥ tick_size. `max_spread` default = 0.10.
- Final quotes: `ask = p_r + spread/2`, `bid = p_r - spread/2`, each rounded to legal tick.
- If inventory ≥ max_inventory_units × trade_unit, bid_volume is set to 0 (no more buying).

### 4.4 Fill simulation (`_match_side`)

The fill model determines whether the agent's resting limit orders would have been executed, using the historical/simulated trade data:

For the agent's **ask** quote at price `p`:
1. **Crossing**: If `p ≤ current_bid`, it's a market-crossing order → instant fill at bid (taker fill, no maker rebate).
2. **Better trades exist**: If any historical trade happened at price > p (more aggressive), the agent's order would have been filled first → fill at `p`.
3. **Same-price trades**: If trades happened exactly at `p`, fill probability = `trade_volume / (trade_volume + level_depth)`. This models queue priority — if the level is thick, the agent is far back in the queue.

Symmetric logic for bid quotes.

### 4.5 Reward function

The reward at each step combines three components:

```
R = dampened_PnL + trading_PnL - inventory_penalty
```

**Dampened PnL** (DP): `DP = dampened_pnl_weight * (ΔPnL - max(0, η*ΔPnL))` where `ΔPnL = value_t - value_{t-1}` and `value = cash + inventory*mid`. The η=0.5 parameter means positive ΔPnL is halved — this discourages speculative inventory holding that happens to profit from price moves.

**Trading PnL** (TP): `TP = trade_reward_weight * Σ fill.volume * (mid - fill.price)`. This rewards the price advantage of each fill relative to mid. A buy below mid or a sell above mid earns positive TP.

**Inventory Penalty** (IP): `IP = inventory_penalty_weight * zeta * (inventory / trade_unit)²`. Quadratic penalty that grows quickly with position size. `zeta` default = 0.01.

### 4.6 Episode termination

At episode end, the agent's remaining position is forcibly closed at the bid (if long) or ask (if short). This closing trade incurs the full crossing cost and is included in the final PnL.

### 4.7 Performance optimization

All latent DataFrame columns (`latent_alpha`, `regime_shift`, `event_actor`, `maker_agent`, `queue_pressure`, `top_imbalance`, `vol_state`) are lazily converted to numpy arrays on first access via `_ensure_log_caches()`. This eliminates per-step `.iloc` calls that were causing ~50x slowdown. The `_agent_state` method similarly reads `ask1_volume` / `bid1_volume` directly from the `day.lob` numpy array (columns 1 and 3) instead of using `.iloc` on the ask/bid DataFrames.

---

## 5. The Model (`models.py`)

### 5.1 Attn-LOB backbone

Input: `(batch, 50, 40)` normalized LOB tensor.

Architecture (following the paper's CNN-Attention design):
1. **Spatial convolutions**: 9 conv layers that reduce the 40-column LOB to a 1-wide feature map while preserving the 50-step temporal dimension. Output: `(batch, 32, 50, 1)`.
2. **Inception module**: Three parallel branches (3×1 conv, 5×1 conv, maxpool+1×1 conv), each producing 64 channels. Concatenated → `(batch, 192, 50)`.
3. **Temporal projection**: Linear 192→160, producing `(batch, 50, 160)`.
4. **Multi-head self-attention**: 10 heads, key_dim=16. Query is the last timestep only → attends over all 50 timesteps. Output: `(batch, 1, 160)`.
5. **Output projection**: Linear 160→64.

Output: 64-dimensional feature vector.

### 5.2 Pretraining

The backbone is pretrained on 3-class mid-price direction prediction (up/flat/down). A linear head is attached: 64→3. Trained with CrossEntropyLoss, selected by best validation F1. The backbone weights are saved and loaded for RL training.

### 5.3 Actor-Critic

`SharedStateEncoder` combines the backbone output (64-dim) with the flat features (48-dim) via a 2-layer MLP: Linear(112, 128) → LeakyReLU → Linear(128, 128) → LeakyReLU.

`ContinuousActorCritic` has:
- **Alpha head**: Linear(128, 2) → softplus + 1.0 → α parameter of Beta distribution
- **Beta head**: Linear(128, 2) → softplus + 1.0 → β parameter of Beta distribution
- **Value head**: Linear(128, 1) → scalar value estimate

Actions are sampled from `Beta(α, β)` which naturally lives in [0, 1]. During evaluation, the distribution mean is used (deterministic).

---

## 6. PPO Training (`rl.py`, `train.py`)

### 6.1 Rollout collection

For each epoch, `ppo_rollouts_per_epoch` episodes are collected by cycling through (env, span) pairs. Each step stores: lob, flat, action, logprob, value estimate, reward, done flag.

### 6.2 GAE computation

After rollouts, Generalized Advantage Estimation is computed:
```
δ_t = r_t + γ * V(s_{t+1}) * (1-done) - V(s_t)
A_t = δ_t + γ * λ * (1-done) * A_{t+1}
returns = advantages + values
```
With `γ=0.99`, `λ=0.95`. Advantages are optionally normalized (mean-subtracted, std-divided).

### 6.3 Policy updates

For each of `ppo_updates` passes (default 2):
- Shuffle all rollout data, split into minibatches of size `ppo_minibatch_size` (default 256)
- For each minibatch:
  - Compute new log_prob and value from current model
  - Policy loss: clipped PPO objective with `clip=0.2`
  - Value loss: 0.5 * MSE(value, returns)
  - Entropy bonus: `-1e-3 * entropy` (encourages exploration)
  - Total loss: `policy_loss + value_loss - entropy_bonus`
  - Gradient clip norm: 1.0

### 6.4 Model selection

If `ppo_select_best_model` is True, after each epoch the model is evaluated on validation days. The model with the best `ppo_selection_metric` (default "pnl_mean") is saved. Final model is the best, not the last.

---

## 7. Baselines (`baselines.py`)

### 7.1 Avellaneda-Stoikov (AS)

The classical optimal market-making model. Quotes:
```
reservation = mid - inventory_units * γ * σ²*(T-t)
spread = γ*σ²*(T-t) + (2/γ)*ln(1 + γ/κ)
```
Where `γ` = risk aversion, `κ` = fill decay parameter, `σ²` = step variance.

Calibration (`calibrate_avellaneda_stoikov`):
- `step_variance`: empirical variance of mid-price diffs on training data
- `κ`: estimated from exponential fit of fill probability vs distance from touch
- `γ`: chosen so that max inventory skew is ~1.5 ticks
- `base_spread`: `max(tick, min(max_spread, 2/κ))`

AS serves as the **health check** for the simulator. If AS has deeply negative PnL, the simulator is producing an environment where passive market-making is structurally unprofitable — meaning the RL agent can't learn useful behavior either.

### 7.2 Fixed level

`FixedLevelPolicy(level=k)` always quotes at the k-th level of the book. Fixed_1 quotes at the touch (tightest spread, most fills, most adverse selection). Fixed_2 quotes one level deeper (wider spread, fewer fills, less adverse selection).

---

## 8. Config (`config.py`)

### 8.1 Key hyperparameters to tune

**Simulator calibration** (affect the market microstructure):
- `price_noise_scale` (0.004): Controls daily price range. Higher → more volatile market.
- `market_order_impact_scale` (0.95): Global multiplier on how much market orders move fair_value.
- `market_order_alpha_impact` (0.0004): How much each market order's fair_value shift depends on the signal. **This is the main "toxicity" knob** — higher values make informed flow more poisonous to makers. The value 0.0004 was found through grid search to produce an environment where AS is slightly profitable.
- `market_order_tick_impact` (0.0015): Base per-trade fair_value shift independent of signal.
- `touch_replenish_fraction` (0.6): What fraction of consumed touch liquidity is immediately restored for noise trades. 0 = no replenishment (broken), 1.0 = full replenishment (touch never moves from noise). 0.6 is the calibrated value.
- `informed_taker_rate_scale` (1.0): Multiplier on how often informed takers fire. Higher → more adverse selection.
- `noise_taker_rate_scale` (1.0): Multiplier on noise taker frequency. Higher → more harmless flow → easier for makers.

**Action space** (affect what the RL agent can do):
- `max_bias` (0.05): Maximum reservation-price offset from mid. At ¥12.5 this is 0.4% of price; at ¥135 it's 0.037%. Consider making this price-proportional if cross-stock consistency matters.
- `max_spread` (0.10): Maximum quoted spread. 10 ticks for ¥12.5 stock, 10 ticks for all stocks.

**Reward weights** (affect what the agent optimizes):
- `dampened_pnl_weight` (1.0): Weight on the dampened PnL component.
- `trade_reward_weight` (1.0): Weight on trading PnL (price advantage per fill).
- `inventory_penalty_weight` (1.0): Weight on quadratic inventory penalty.
- `eta` (0.5): Dampening parameter. Higher → more punishment of speculative holding profits.
- `zeta` (0.01): Inventory penalty scaling. Higher → agent keeps inventory closer to zero.

### 8.2 Mode defaults

- `smoke`: 4 days, 500 events/day, 1 epoch everything. For syntax/pipeline testing only.
- `medium`: 8 days, 20k events/day, 4 epochs. For development/iteration.
- `full`: 21 days, 60k-120k events/day, 6-10 epochs. For final results.

---

## 9. Report and Diagnostics (`report.py`)

The report evaluates PPO on test data and compares against AS, Fixed_1, Fixed_2. Key metrics:

- **PnL**: Total profit/loss per episode. The most important number.
- **ND-PnL**: PnL / avg_spread. "How many spreads did the agent capture?"
- **PnLMAP**: PnL / mean_abs_position. "How much profit per unit of risk?"
- **Sharpe**: mean(PnL) / std(PnL). Stability of profits.
- **Fill rate**: Fraction of steps where at least one fill occurred.
- **bias_alpha_corr**: Correlation between the agent's quote bias and the latent signal. Positive = the agent learned to skew quotes in the direction of the signal (good).

Added diagnostics:
- **spread_mean / spread_std / spread_gt1_frac**: Spread distribution during evaluation.
- **vol_state_mean / vol_state_std**: Volatility clustering behavior.

---

## 10. Current Calibration Status

As of the latest version, with default config:

| Metric | Value | Target |
|---|---|---|
| Mean spread (ticks) | 1.36 | 1.2-2.0 ✅ |
| Spread > 1 tick fraction | 36% | 15-40% ✅ |
| Daily price range | 0.4-1.1% | 0.5-2% ✅ |
| Informed trade share | 34% | 25-40% ✅ |
| Noise-buy adverse drift (20 events) | +0.009 ticks | ≈0 ✅ |
| Informed-buy adverse drift (20 events) | +0.196 ticks | >0 ✅ |
| AS baseline PnL | +30 | >0 ✅ |
| Fixed_1 PnL | +16 | >0 ✅ |
| Env step speed | ~11k steps/sec | fast ✅ |

---

## 11. Known Issues and Next Steps

### 11.0 LOB-observable alpha (new knobs, default OFF)

The Euler run comparing `piroth` vs `anja_simulations` showed that AS beats PPO by a large margin in both simulators. Root cause: the latent `signal` drives *trade rates* and *fair value* but does not tilt the *book shape* before the informed trades arrive. Attn-LOB's 50-event window sees no leading indicator and cannot beat AS.

Three mechanisms (all gated, default off) add LOB-observable alpha:

- `lob_leak_strength` (default 0.0): Amplitude of asymmetric maker_add/cancel events in `_event_weights` + asymmetric touch depth in `_ensure_side_depth`. When the signal is positive, maker_add_bid fires more often, maker_cancel_bid less, and touch-bid depth is thicker (inversely for ask). Creates LOB imbalance that leads price.
- `signal_threshold_for_lob_leak` (default 0.5): Activation threshold. The leak is inactive for `|signal| < threshold`, so benign regimes stay benign.
- `informed_hawkes_alpha` (default 0.0): Self-excitation — each informed trade boosts the probability of the next informed trade through an exponentially-decaying kernel `self.informed_clock`. Default decay `informed_hawkes_decay=0.97` gives ~30-event half-life. Enables trade clustering that OSI/RSI features can pick up.

Rollout order is conservative: enable Hawkes first, then LOB-leak, then re-check invariants.

**New diagnostic metric: `lob_imbalance_future_return_corr`**. See `diagnostics.py`. Target band at horizon=50: **0.10–0.15** (enough signal for Attn-LOB to extract, not so much that the env becomes trivial). Run:

```
python -m anja_simulations.diagnostics --data-dir <path> --symbol 000001 --days 3
```

The script also reports trade-sign ACF, |return| ACF (volatility clustering), spread-depth correlation, log-return kurtosis, and informed/noise trade shares.

### 11.1 Experimental Status (as of 2026-04-22)

**Objective:** Improve PPO's ability to beat the Avellaneda-Stoikov baseline through LOB-observable alpha mechanisms. The paper (Guo et al.) shows PPO Sharpe ~12.3 vs AS ~0.74 on real SHE data; current synthetic env has PPO ~0.67–1.25 vs AS ~2.0–4.0 (reversal).

**Medium-mode Euler runs completed:**

| Job | Config | PPO Sharpe | AS Sharpe | PPO trades/ep | test_f1 | lob_imb[50] | Status |
|---|---|---|---|---|---|---|---|
| 64422018 | leak=0.3, hawkes=0.1 | 1.25 | 4.04 | 2.22 | 0.109 | 0.048 | ✓ Best PPO |
| 64427654 | leak=0.5, hawkes=0.1 | 1.19 | 3.74 | 4.44 | 0.1198 | 0.0643 | ~ marginal gain |
| 64430005 | leak=0.5, hawkes=0.1, alpha=5e-4 | 0.95 | 2.0245 | 1.33 | 0.01 | — | ✗ regression |

**Key findings:**
1. **LOB-leak strength shows diminishing returns.** Going from leak=0.3→0.5 bought corr[50] from 0.048→0.0643 (+33%) but PPO Sharpe actually declined 1.25→1.19. Suggests 0.048–0.064 is still below the threshold where RL benefits materially (target 0.10–0.15).
2. **Pretrain overfitting is the binding constraint.** Test F1 of 0.01–0.12 (vs target ~0.35) means the backbone learns per-day artifacts that don't generalize. Tuning pretrain_alpha blindly is unproductive — alpha=5e-4 made things worse without measurement first.
3. **BUG fixed in report.py:** Baselines were evaluated without `env.set_eval_context()`, causing non-deterministic fills while PPO used seeded fills. Invalidated comparisons. Single-line fix applied (line 22).

**Current bottleneck analysis:**
- Leak strength is plateauing; more leakage won't solve the problem (already tried 0.3→0.5).
- Pretrain overfitting (test_f1 too low) means PPO gets bad features, so it opts out of trading.
- Two paths forward:
  - **Path A (abandoned):** Tune pretrain_alpha without measurement — experiment 64430005 shows this is a trap.
  - **Path B (recommended):** Jump to MODE=full. With 5× the training data per day, generalization should improve organically. Full mode is the real target anyway.

### 11.2 Next Steps (recommended)

**Immediate action:** Submit MODE=full with `LOB_LEAK_STRENGTH=0.5`, `INFORMED_HAWKES_ALPHA=0.1`, `PRETRAIN_ALPHA=1e-5` (keep default, skip alpha tuning). Estimated wall-clock time: 8–12 hours on one GPU.

**Post-submission (after full run completes):**
1. Run diagnostics: `python -m anja_simulations.diagnostics --data-dir <path> --symbol 000001 --horizons 10 50 200`
2. **Gate 1:** Check `lob_imbalance_future_return_corr[50]` ≥ 0.10. If not, signal is still too weak for RL — may need to reconsider leak mechanism or increase decay (make signal more persistent).
3. **Gate 2:** Check PPO Sharpe > AS Sharpe (the paper's inversion). If not, the backbone is still not extracting useful signal — may need architecture changes or the leak mechanism is fundamentally flawed.
4. If both gates pass: the synthetic env is paper-like, mission accomplished. If not, debug:
   - If gate 1 fails (weak corr): consider increasing `INFORMED_HAWKES_ALPHA` to 0.2 or adding a fourth mechanism (momentum state variable?).
   - If gate 2 fails (PPO still trails AS): could try skipping pretrain entirely (use raw features) or increasing entropy bonus to force exploration.

### 11.3 Known issues (older, lower priority)

1. **`max_bias` / `max_spread` are not price-scaled**: For 000858 at ¥135, `max_bias=0.05` is only 0.037% of price, while for 000001 at ¥12.5 it's 0.4%. The agent has effectively much less room to adjust for expensive stocks. Consider `max_bias = 0.05 * (base_price / 12.5)` or a similar scaling.

2. **Volatility clustering is gentle**: vol_state range is 0.6-1.8 but in practice stays near 1.2-1.5 with std ~0.03-0.15. The agent may not get enough variance to learn truly distinct behavior for volatile vs calm periods. Can increase `vol_shock` coefficient or widen the GARCH-like response.

3. **Per-day LOB normalization breaks generalization** (data.py `_paper_normalize_lob`): Volumes are normalized by day-specific maxima, so "volume=0.5" on day 1 means something different than "volume=0.5" on day 10. This contributes to pretrain test-set overfitting. Could use corpus-wide max instead of per-day max.

4. **Cross-day independence**: Each day is generated independently with no carry-over of book state or fair value. This is a simplification — real markets have overnight gaps and opening dynamics.

---

## 12. Euler Submission Commands

**Full suite (mode=full) on anja_simulations with LOB-observable alpha enabled:**

From local PowerShell:
```powershell
$ts = Get-Date -Format yyyyMMdd_HHmmss
C:\Windows\System32\OpenSSH\ssh.exe -i C:\Users\anjic\.ssh\id.eddsa apetric@euler.ethz.ch "bash -lc 'cd /cluster/home/apetric/mlfcs-gapa/simulation && RUN_NAME=anja_full_${ts} MODE=full SYMBOLS=000001 LOB_LEAK_STRENGTH=0.5 INFORMED_HAWKES_ALPHA=0.1 OUTPUT_ROOT=/cluster/scratch/apetric/artifacts_anja DATA_DIR=/cluster/scratch/apetric/data/anja_full_${ts} ACCOUNT=ls_math bash cluster/submit_anja_suite.sh'"
```

From Euler login node:
```bash
cd /cluster/home/apetric/mlfcs-gapa/simulation
TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME=anja_full_${TS} \
MODE=full \
SYMBOLS=000001 \
LOB_LEAK_STRENGTH=0.5 \
INFORMED_HAWKES_ALPHA=0.1 \
OUTPUT_ROOT=/cluster/scratch/apetric/artifacts_anja \
DATA_DIR=/cluster/scratch/apetric/data/anja_full_${TS} \
ACCOUNT=ls_math \
bash cluster/submit_anja_suite.sh
```

**Monitoring:**
```bash
squeue -u apetric
sacct -j <JOBID> --format=JobID,JobName%25,State,Elapsed,ExitCode -n -P
tail -n 200 /cluster/home/apetric/mlfcs-gapa/simulation/cluster/logs/anja-suite-<JOBID>.out
```

**After completion, download and run diagnostics locally:**
```bash
python -m anja_simulations.diagnostics \
  --data-dir /path/to/euler_downloads/anja_full_<timestamp> \
  --symbol 000001 --horizons 10 50 200 --output report.json
```
