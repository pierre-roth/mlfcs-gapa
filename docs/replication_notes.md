# Replication Notes: Market Making with Deep Reinforcement Learning from Limit Order Books

This note summarizes what must be replicated from Guo, Lin, and Huang, "Market Making with Deep Reinforcement Learning from Limit Order Books", and what must be adapted because the original Shenzhen Stock Exchange order and trade data is unavailable.

## Primary Sources

- Local paper source: `paper/paper.tex`
- Official demo repository: https://github.com/imTurkey/Market-Making-with-Deep-Reinforcement-Learning-from-Limit-Order-Books
- Paper arXiv page: https://arxiv.org/abs/2305.15821
- Avellaneda-Stoikov baseline: https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf
- Spooner et al. RL reward baseline: https://arxiv.org/abs/1804.04216
- Zhong et al. LOB-RL baseline: https://www.ijcai.org/proceedings/2020/0615.pdf
- Lim and Gorse inventory RL baseline: https://discovery.ucl.ac.uk/10116730/1/RLforHFMM.pdf
- FI-2010 benchmark data reference: https://arxiv.org/abs/1705.03233
- LOBSTER data structure reference: https://data.lobsterdata.com/info/DataStructure.php
- LOBSTER sample-style data reference: https://huggingface.co/datasets/totalorganfailure/lobster-data

## Exact Paper Target

The target is a historical event-replay market-making system.

Data:

- Shenzhen Stock Exchange orders and trades, November 2019.
- 21 trading days.
- Stocks:
  - Ping An Bank, SZ.000001
  - Wuliangye Yibin, SZ.000858
  - Hikvision, SZ.002415
- Roughly 5,000,000 LOB samples.
- Pretraining split:
  - first half of month: 10 train days
  - second half: 11 test days
  - 20 percent of training data used for validation
- Stable trading windows:
  - 10:00-11:30
  - 13:00-14:30
- LOB depth: 10 levels.
- LOB snapshot vector: `{ask_price_i, ask_volume_i, bid_price_i, bid_volume_i}_{i=1..10}`.
- Input window: `T = 50`, shape `(50, 40)`.

Pretraining:

- Task: future mid-price direction classification.
- Classes: up, stationary, down.
- Horizon: `k = 10` events.
- Threshold: `alpha = 1e-5`.
- Label formula:
  - rolling past mean and future mean are compared by relative change.
  - up if relative change exceeds `alpha`.
  - stationary if in `[-alpha, alpha]`.
  - down if below `-alpha`.
- Normalization:
  - price levels transformed relative to mid-price.
  - volumes max-normalized.
  - paper text mentions z-normalization after stationary price transform, but the official demo code uses relative-to-mid price normalization.

Market state:

- Dynamic state has 24 features:
  - RV over 300s, 600s, 1800s: 3 features.
  - RSI over 300s, 600s, 1800s: 3 features.
  - OSI over 10s, 60s, 300s, for market, limit, and withdrawal/cancel events, each by volume and count: 18 features.
- Agent state:
  - inventory
  - remaining time
  - official code repeats each into a 24-dimensional vector: inventory repeated 12 times, normalized time repeated 12 times.

Simulator:

- Historical event replay.
- Agent submits one bid and one ask limit order per decision.
- Minimum trade unit: 100.
- Transaction cost: 0.
- Negative cash and inventory allowed.
- Inventory cap: `omega * minimum_trade_unit`, with `omega = 10`, so cap is 1000 shares.
- If inventory exceeds cap, quoting in the direction that worsens inventory is prohibited.
- Agent order volume is small, so no market impact is applied.
- Episode length: 2000 events, about 3-5 minutes.
- Agent value starts at 0 each episode.
- At episode end, inventory is closed with market orders.
- Test data can be traversed only once.
- Training data can be traversed one or more times.

Fill model:

- Paper says orders execute only when real historical order arrivals occur.
- If bid crosses current best ask, execute at counterparty ask.
- If ask crosses current best bid, execute at counterparty bid.
- If passive quote is better than touched historical trade price, fill fully.
- If passive quote equals touched historical trade price, official code assumes the agent is at the back of queue and fills probabilistically:
  - probability = historical traded volume at that price / (historical traded volume at that price + displayed LOB depth at that level).

Action spaces:

- Discrete:
  - Paper says 8 actions, actions 0-7.
  - Actions quote spread/bias pairs; action 7 closes position with market order.
  - Official code has a likely inconsistency: `num_values=5` but action logic handles 0-7.
- Continuous:
  - Paper says `A1, A2` in `[0, 1]`.
  - Official Tensorforce action spec says min -1, max 1, but comments and formulas use [0, 1].
  - `delta = A1 * max_bias`, with `max_bias = 0.05`.
  - `spread = A2 * max_spread`, with `max_spread = 0.1`.
  - reservation price:
    - paper: `p_r = p_m - sign(inventory) * delta`
    - official code equivalent: lower reservation when long, raise it when short.
  - ask/bid:
    - ask = reservation + spread / 2
    - bid = reservation - spread / 2
  - legal price rounding:
    - ask rounded up to cent
    - bid rounded down to cent

Reward:

- Paper target:
  - `R_t = DP_t + TP_t - IP_t`
  - `DP_t = DeltaPnL_t - max(0, eta * DeltaPnL_t)`
  - `TP_t = trade_volume * (mid_price - trade_price)`, with buy volume positive and sell volume negative.
  - `IP_t = zeta * inventory_t^2`
  - Hyperparameters: `eta = 0.5`, `zeta = 0.01`.
- Official code caveat:
  - Continuous env sets `theta = 0.01`, `eta = 0.9`.
  - It computes dampened PnL, matched/trading PnL, and inventory punishment, but returns `pnl - spread_punishment`; the paper reward is commented out.
  - Discrete env also returns raw `pnl`, with hybrid reward commented out.
  - For a faithful paper replication, implement the paper reward, not the demo code return value. For a demo-code replication, preserve the discrepancy as an ablation.

RL algorithms:

- C-PPO:
  - Continuous action space.
  - PPO.
  - Paper reports it as the best method.
  - Official code uses Tensorforce PPO, batch size 32, learning rate default 1e-3 in agent factory, discount 0.99.
  - Main config default learning rate is 1e-4, but PPO factory default is 1e-3; CLI config likely overrides to 1e-4 when passed.
- D-DQN:
  - Discrete action space.
  - Dueling DQN.
  - Official code uses Tensorforce dueling DQN, memory 200000, batch size 32, learning rate 1e-4, horizon 1, discount 0.99, parallel interactions 10.
- Official training loop:
  - train days: 20191101, 20191104, 20191105, 20191106, 20191107, 20191108, 20191111, 20191112
  - test days: 20191113, 20191114, 20191115, 20191118, 20191119, 20191120, 20191121, 20191122, 20191125, 20191126, 20191127, 20191128, 20191129
  - `num_step_per_episode = 2000`
  - `n_train_loop = 5`

Network:

- Attn-LOB encoder:
  - Input: `(T, 40, 1)`, with T=50.
  - Initial convolution stack:
    - Conv2D 32 filters `(1, 2)`, stride `(1, 2)`, LeakyReLU
    - two Conv2D 32 filters `(4, 1)`, same padding, LeakyReLU
    - Conv2D 32 filters `(1, 5)`, stride `(1, 5)`, LeakyReLU
    - two Conv2D 32 filters `(4, 1)`, same padding, LeakyReLU
    - Conv2D 32 filters `(1, 4)`, LeakyReLU
    - two Conv2D 32 filters `(4, 1)`, same padding, LeakyReLU
  - Inception module:
    - branch 1: Conv2D 64 `(1,1)` then Conv2D 64 `(3,1)`
    - branch 2: Conv2D 64 `(1,1)` then Conv2D 64 `(5,1)`
    - branch 3: MaxPool `(3,1)` then Conv2D 64 `(1,1)`
    - concatenate into 192 channels
  - Attention:
    - reshape to `(T, 192)`
    - query is last timestep only
    - MultiHeadAttention with 10 heads, key_dim 16, output_shape 64
    - flatten to 64-dimensional LOB embedding
  - Pretrain head:
    - Dense 3 softmax.
  - RL head:
    - concatenate LOB embedding, agent state, market state
    - Dense 64 LeakyReLU.
- Official code caveat:
  - `get_model` appears to append `agent_state` instead of `market_state` in the market state branch. We should fix this for paper-faithful replication and optionally keep the bug as a demo-code ablation.

## Figure-by-Figure Replication Requirements

The paper has four figures. These are not decorative; each one implies a concrete artifact we need to reproduce.

### Figure 1: Attn-LOB Architecture

File: `paper/figure_1.png`

What the figure fixes exactly:

- Input tensor is `50 x 40`.
- Three initial convolution reductions transform the width dimension:
  - `40 -> 20` by Conv2D `32 @ 1x2`, stride `1x2`.
  - `20 -> 4` by Conv2D `32 @ 1x5`, stride `1x5`.
  - `4 -> 1` by Conv2D `32 @ 1x4`.
- Time dimension remains 50 throughout the convolution stack.
- Inception module has three branches:
  - branch A: Conv2D `64 @ 1x1`, then Conv2D `64 @ 3x1`.
  - branch B: Conv2D `64 @ 1x1`, then Conv2D `64 @ 5x1`.
  - branch C: MaxPool `3x1`, then Conv2D `64 @ 1x1`.
- Branch outputs concatenate to 192 channels.
- Multi-head self-attention attends from the last timestep query over the 50-step sequence.
- Attention output is 64-dimensional.
- LOB state embedding is concatenated with:
  - agent state
  - dynamic state
- An MLP maps this combined state to the action space.
- For pretraining, the 64-dimensional LOB state goes to a 3-class softmax: up, stationary, down.

Implementation implication:

- The PyTorch encoder should have a `return_attention_weights` option so Figure 3 can be reproduced.
- The encoder should expose the 64-dimensional latent vector separately from the classifier head.
- Unit tests should assert tensor shapes after each stage because width reduction is easy to get wrong.

### Figure 2: Latency Experiments

File: `paper/figure_2.png`

The paper does not publish the underlying numeric latency table. Only the plotted curves are available. This means exact paper-number replication of the latency sweep is impossible unless:

- the authors provide the figure data, or
- we digitize the plotted figure and label those values as approximate, or
- we rerun the experiments and generate our own exact numbers under our data.

Observed latency settings from the x-axis:

- approximately `1, 5, 10, 20, 50, 100` events.

Panel layout:

- (a) C-PPO
- (b) D-DQN
- (c) AS
- (d) Random
- (e) Fixed

Metrics per panel:

- blue: ND-PnL
- orange: PnLMAP
- green: Profit Ratio

Qualitative behavior to reproduce:

- C-PPO remains profitable and degrades slowly at high latency.
- C-PPO appears strongest around latency near 10 events.
- The paper explains this by noting the pretraining horizon is also 10.
- D-DQN is less stable but remains adaptive.
- AS, Random, and Fixed degrade strongly with latency.
- Fixed becomes sharply negative on PnLMAP and ND-PnL as latency rises.

Experiment implementation:

- Add a `latency_events` parameter to the environment.
- Use the delayed state `state(t - latency_events)` while executing fills at event `t`.
- Sweep latency values `[1, 5, 10, 20, 50, 100]`.
- Run the same trained policy under each latency value, unless intentionally training latency-specific agents.
- Save exact numeric CSV for every run:
  - `method`
  - `stock_or_dataset`
  - `seed`
  - `latency_events`
  - `nd_pnl`
  - `pnl_map`
  - `profit_ratio`
  - `sharpe`
  - `mean_inventory`
  - `mean_abs_inventory`
  - `runtime_ms_per_step`

Approximate visual reading from Figure 2, not exact:

| Method | Latency | ND-PnL trend | PnLMAP trend | Profit Ratio trend |
|---|---:|---|---|---|
| C-PPO | 1-100 | around high 20s to high 30s, then low 20s | peaks near 10, then declines | low single digits, slowly down |
| D-DQN | 1-100 | rises to around 8-9 near 20-50, then declines | high at very low latency, drops sharply, partial recovery | unstable, low to high single digits |
| AS | 1-100 | low positive, gently declines | positive, declines steadily | positive, declines steadily |
| Random | 1-100 | moves from positive to negative | moves from positive to strongly negative | moves from positive to negative |
| Fixed | 1-100 | positive to strongly negative | positive to very negative | mildly declines |

Do not treat these as source numbers. The exact source numbers are absent.

### Figure 3: Attention Visualization

File: `paper/figure_3.png`

Purpose:

- Show that the attention layer focuses on recent LOB changes.
- Compare stable markets to rapidly changing markets.

Required plot structure:

- Four examples:
  - two stable market examples
  - two rapidly changing market examples
- Each example has:
  - top bar plot: attention score over 50 timestamps
  - lower heatmap: LOB volume and price features over 50 timestamps
- The heatmap y-axis separates:
  - volume features: ask and bid depth
  - price features: ask and bid levels

Observed pattern:

- Stable cases: most attention mass is on the last 5-10 timestamps.
- Rapidly changing cases: attention still emphasizes recent timestamps, but also assigns visible mass to earlier changes around timestamps 8-10, 18-20, 28-38, or 42-49 depending on event structure.

Implementation implication:

- Save attention weights from the same trained C-PPO encoder used in the policy, not from a separately trained classifier unless explicitly stated.
- For exact visual replication, choose example windows from test data where:
  - stable example: mid-price nearly flat, depth stable until late-window perturbation.
  - rapid example: visible shifts in price/depth across the window.
- Store the selected event IDs so the figure can be regenerated.

### Figure 4: Historical Decision Making

File: `paper/figure_4.png`

Purpose:

- Demonstrate learned market-making behavior, not just aggregate metrics.

Required plot structure:

- Top panel:
  - black: mid-price or price
  - blue: agent value on right y-axis
  - red: bid quote
  - green: ask quote
- Bottom panel:
  - inventory over time
  - dashed green zero line
- X-axis:
  - time in seconds, roughly 4500 to 8500 in the source figure.

Observed behavior:

- First half:
  - price oscillates around a band.
  - agent earns value while opening and closing inventory.
  - when inventory is negative, bid quote moves closer to mid-price to encourage buying and neutralize position.
- Second half:
  - price falls sharply.
  - bid and ask quotes move farther from mid-price.
  - inventory stays near zero for long stretches, avoiding trend exposure.
- Value rises from around 145 to around 250, with a short drawdown near the end.

Implementation implication:

- The environment logger must record at every decision event:
  - timestamp or elapsed seconds
  - mid_price
  - bid_quote
  - ask_quote
  - cash
  - inventory
  - marked_value
  - trade_price
  - trade_volume
  - action vector
  - reservation price
  - spread
- The plot should be generated from a held-out episode, not training data.
- Store the episode ID and seed.

## Exactness Policy

Use three tiers when reporting results:

1. Paper-exact:
   - values copied directly from the paper tables.
   - includes pretraining, overall results, ablation table, and runtime table.
2. Figure-estimated:
   - values approximated from plots because the table is not published.
   - latency plot falls into this category unless authors provide source data.
3. Replication-exact:
   - exact values produced by our own code, saved to CSV/Parquet and reproducible by seed/config.

The final project should never present figure-estimated numbers as paper-exact.

## Missing Information Needed for a Truly Exact Replication

The paper and official demo repository do not provide all details required for a byte-for-byte exact replication. Missing or ambiguous items:

- Original Shenzhen order/trade data.
- Exact train/validation split details for the stated 10 training days.
- Exact random seeds and number of independent runs behind `mean +/- std`.
- Exact PPO hyperparameters if the paper used values different from the official factory defaults.
- Exact Dueling DQN target-network update schedule, epsilon schedule, and replay warmup.
- Exact AS calibration method for `gamma`, `sigma`, and `kappa`.
- Exact random quoting distribution.
- Exact Fixed_1/2/3 fill and inventory-block behavior.
- Exact Sharpe definition:
  - per episode
  - per day
  - annualized or not
- Numeric data behind Figure 2 latency curves.
- Whether paper tables used the hybrid reward as written or the official demo code's raw-PnL reward.

Required mitigation:

- Build paper-faithful implementation first.
- Keep a `paper_ambiguities.md` or config notes section tracking every assumption.
- Save all run configs, seeds, metrics, and environment versions.

## Reported Numbers

### Pretraining Table

| Model | Precision | Recall | F1 | Parameters | Input |
|---|---:|---:|---:|---:|---|
| FC-LOB | 0.6315 | 0.5419 | 0.5660 | 256,064 | 4000 x 1 |
| Conv-LOB | 0.5851 | 0.5230 | 0.4984 | 172,320 | 1024 x 40 |
| DeepLOB | 0.7856 | 0.6699 | 0.7118 | 139,168 | 100 x 40 |
| Attn-LOB | 0.7663 | 0.7019 | 0.7284 | 176,320 | 50 x 40 |

### Overall Results

Ping An Bank:

| Method | ND-PnL x1e5 | PnLMAP | PR x1e-4 | Sharpe |
|---|---:|---:|---:|---:|
| C-PPO | 9.3 +/- 0.7 | 117.2 +/- 3.8 | 5.0 +/- 0.1 | 12.3 +/- 0.8 |
| D-DQN | 7.0 +/- 1.7 | 8.6 +/- 2.2 | 3.5 +/- 0.2 | 1.3 +/- 0.7 |
| Inv-RL | 0.3 +/- 0.1 | 24.7 +/- 4.2 | 4.3 +/- 0.9 | 1.3 +/- 0.3 |
| LOB-RL | 1.1 +/- 0.5 | 1.3 +/- 0.5 | 2.8 +/- 1.4 | 0.2 +/- 0.3 |
| AS | 0.49 | 4.75 | 4.22 | 0.74 |
| Random | 0.39 | 0.81 | 0.93 | -0.19 |
| Fixed_1 | 2.63 | 4.70 | 1.28 | -0.01 |
| Fixed_2 | 0.97 | 2.03 | 9.97 | 0.21 |
| Fixed_3 | 0.25 | 1.41 | 21.58 | 0.31 |

Wuliangye:

| Method | ND-PnL x1e5 | PnLMAP | PR x1e-4 | Sharpe |
|---|---:|---:|---:|---:|
| C-PPO | 19.8 +/- 1.8 | 630.6 +/- 85.5 | 2.8 +/- 0.5 | 2.2 +/- 0.7 |
| D-DQN | 11.0 +/- 3.7 | 28.4 +/- 12.8 | 0.9 +/- 0.1 | -0.5 +/- 0.1 |
| Inv-RL | 3.8 +/- 0.4 | 70.2 +/- 23.0 | 0.7 +/- 0.1 | -1.3 +/- 0.2 |
| LOB-RL | 1.8 +/- 1.1 | 6.8 +/- 5.0 | 0.2 +/- 0.1 | -0.7 +/- 0.3 |
| AS | 3.14 | 19.61 | 3.93 | 0.17 |
| Random | 0.86 | 3.33 | 0.15 | -0.81 |
| Fixed_1 | -3.12 | -10.72 | -0.12 | -4.88 |
| Fixed_2 | 7.66 | 26.57 | 2.36 | 0.55 |
| Fixed_3 | 3.84 | 13.36 | 3.89 | 0.36 |

Hikvision:

| Method | ND-PnL x1e5 | PnLMAP | PR x1e-4 | Sharpe |
|---|---:|---:|---:|---:|
| C-PPO | 16.0 +/- 3.3 | 313.6 +/- 25.9 | 3.8 +/- 0.6 | 7.1 +/- 0.5 |
| D-DQN | 11.7 +/- 6.8 | 65.2 +/- 10.0 | 10.1 +/- 3.3 | 0.4 +/- 0.1 |
| Inv-RL | 1.4 +/- 0.1 | 52.7 +/- 16.6 | 2.4 +/- 0.4 | 1.0 +/- 0.4 |
| LOB-RL | 1.9 +/- 0.9 | 5.9 +/- 3.9 | 0.7 +/- 0.4 | -0.2 +/- 0.4 |
| AS | 1.57 | 16.39 | 8.10 | 0.65 |
| Random | 2.76 | 6.73 | 1.75 | 0.27 |
| Fixed_1 | 1.43 | 3.71 | 0.24 | -1.69 |
| Fixed_2 | 4.49 | 10.55 | 6.38 | 1.02 |
| Fixed_3 | 1.62 | 4.60 | 10.53 | 0.48 |

### Runtime Table

| Method | Runtime ms/timestep |
|---|---:|
| Random | 10.0 |
| Fixed | 10.9 |
| AS | 19.7 |
| D-DQN inference | 46.7 |
| D-DQN train | 77.5 |
| C-PPO inference | 47.5 |
| C-PPO train | 75.7 |

Paper context: average interval between events is 60-150 ms, so the reported neural-agent inference times were considered acceptable.

### Ablation Table

| Method | ND-PnL x1e5 | PnLMAP | PR x1e-4 | Sharpe |
|---|---:|---:|---:|---:|
| C-PPO | 9.34 | 117.18 | 5.01 | 12.34 |
| C-PPO w/o LOB state | 0.15 | 17.13 | 10.66 | 1.20 |
| C-PPO w/o Attn-LOB | 0.58 | 22.18 | 11.15 | 1.43 |
| C-PPO w/o Dynamic state | 9.12 | 112.52 | 5.05 | 13.32 |
| D-DQN | 6.98 | 8.65 | 3.54 | 1.25 |
| D-DQN w/o LOB state | 4.52 | 6.60 | 1.70 | 2.71 |
| D-DQN w/o Attn-LOB | 6.96 | 7.83 | 3.46 | 1.31 |
| D-DQN w/o Dynamic state | 6.38 | 7.90 | 3.95 | 1.11 |

Interpretation:

- LOB state is essential, especially for C-PPO.
- Attn-LOB matters heavily for C-PPO.
- Dynamic features matter less than LOB state but still contribute.
- D-DQN is less sensitive to replacing Attn-LOB, suggesting the discrete action space may be the bottleneck.

### Latency Experiments

The paper plots latency results in Figure 2 but does not provide the numeric table behind the plot. Exact numeric replication requires either digitizing `paper/figure_2.png` or rerunning experiments.

Qualitative results:

- C-PPO remains robust under latency and peaks around latency near 10 events.
- The paper explains this by the pretraining horizon also being 10 events.
- D-DQN is somewhat robust but less stable than C-PPO.
- AS, Random, and Fixed degrade as latency increases because their parameters are fixed and do not adapt to delayed state.

Approximate latency x-axis values in the figure:

- 1, 5, 10, 20, 50, 100 events.

For our replication, save exact latency sweep output as structured CSV:

- method
- latency_events
- seed
- ND-PnL
- PnLMAP
- profit_ratio
- Sharpe
- mean_inventory
- mean_abs_inventory
- runtime_ms_per_step

## Baseline Definitions

### Fixed Quoting

- Fixed_1, Fixed_2, Fixed_3 quote bid and ask at fixed LOB depths.
- Fixed_1 uses level 1.
- Fixed_2 uses level 2.
- Fixed_3 uses level 3.
- Always quote one unit per side unless inventory cap blocks one direction.
- The paper notes Fixed_3 often has best profit ratio but low ND-PnL/PnLMAP, because wider quotes trade less often but with better price advantage.

### Random

- Paper: randomly quotes ask and bid orders in the five levels of the LOB.
- Related Zhong et al. random baseline: flip one fair coin for best bid and one for best ask each period.
- For faithful paper replication, implement random level selection across levels 1-5 on each side.

### Avellaneda-Stoikov

Use the paper equations:

- reservation price: `r(s, q, t) = s - q * gamma * sigma^2 * (T - t)`
- total spread: `delta_a + delta_b = gamma * sigma^2 * (T - t) + (2/gamma) * log(1 + gamma/kappa)`
- quote symmetrically around reservation price.

Required calibration under substituted data:

- `sigma`: realized mid-price volatility over training data, episode-local or day-local.
- `gamma`: risk aversion grid; select by validation.
- `kappa`: fill intensity slope. Fit execution probability or market-order arrival intensity as a function of quote distance from mid.

If no reliable event-level data is available, use a transparent grid search for gamma/kappa and report it as validation-tuned AS.

### Inv-RL

From Lim and Gorse:

- Uses inventory and remaining time as state.
- Q-learning style formulation.
- Terminal CARA utility helps account for risk aversion and end-of-period inventory.

For our replication:

- Build a small RL baseline using only normalized inventory and time-to-episode-end.
- Use the same action space as C-PPO or the discrete action space, but report the choice clearly.
- To align with the paper table, Inv-RL should not receive LOB state.

### LOB-RL

From Zhong et al.:

- State aggregation based on:
  - bidSpeed
  - askSpeed
  - avgmidChangeFrac
  - invSign
  - cumPnL
- Action space in Zhong et al.:
  - whether to rest at best bid and/or best ask: `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`.
- State size after aggregation: 200.
- Q-table size after action restrictions: 640.

For the Guo et al. paper's LOB-RL baseline, they describe handcrafted features:

- bidSpeed
- askSpeed
- avgmidChangeFrac
- invSign
- cumPnL

Implemented as a tabular Q-learning baseline with those handcrafted features:

- `bidSpeed`: market sell volume exceeds displayed bid depth at levels 1+2.
- `askSpeed`: market buy volume exceeds displayed ask depth at levels 1+2.
- `avgmidChangeFrac`: signed bucket of the change in average mid-price across adjacent recent windows.
- `invSign`: signed bucket of inventory in 100-share lots.
- `cumPnL`: one-bit bucket for cumulative account value below/above threshold.
- Action set is `{(0,0), (0,1), (1,0), (1,1)}` for quoting at best bid and/or best ask.
- The Zhong inventory-side restriction is implemented: when `invSign=-2`, only no quote or bid-only are admissible; when `invSign=+2`, only no quote or ask-only are admissible.

### D-DQN

- Paper's discrete RL agent.
- Uses LOB representation, dynamic state, and agent state.
- Uses discrete action space.
- Dueling DQN algorithm.

Implemented as a compact PyTorch dueling Double DQN:

- discrete Gymnasium replay environment with 8 actions;
- actions 0-6 are the paper/demo tick-offset quote actions;
- action 7 immediately closes inventory with a market order;
- Q-network uses Attn-LOB over the LOB state plus dynamic state and agent state;
- dueling value/advantage heads with Double-DQN target selection;
- target network, replay buffer, epsilon-greedy exploration, and checkpoint saving.

### C-PPO

- Paper's main agent.
- Uses Attn-LOB, dynamic state, agent state, continuous action space, and hybrid reward.
- Implemented with Stable-Baselines3 PPO using a custom Gymnasium environment and custom PyTorch Attn-LOB feature extractor.
- Optional pretrained Attn-LOB encoder checkpoint loading is supported, with a freeze switch for ablations.

## Metrics

Per episode:

- `value = cash + inventory * mid_price`
- `PnL = final_value - initial_value`
- `average_spread = mean(quoted_ask - quoted_bid)` when both quotes are active
- `ND-PnL = PnL / average_spread`
- `MAP = mean(abs(inventory))`
- `PnLMAP = PnL / (MAP + eps)`
- `profit_ratio = PnL / total_buy_notional_or_total_traded_volume`
- `Sharpe = mean(episode_or_daily_returns) / std(episode_or_daily_returns)`

The official code counts volume as buy notional only:

- `volume += max(0, trade_volume * trade_price)`

We should decide whether to reproduce this exactly or use total absolute traded notional. For comparability with paper/demo code, start with buy-notional volume and add a second conventional metric later.

## Data Substitution Plan

Because we do not have the proprietary Shenzhen event/order/trade data, we need a staged data plan.

Stage 1: synthetic replay data

- Goal: validate simulator, fill logic, accounting, reward, metrics, and training loops.
- Generate realistic enough event streams:
  - mid-price random walk with volatility regimes
  - spread in ticks
  - 10-level depth curve
  - market buy/sell events
  - limit/cancel events for OSI
  - matched trade extrema per event for fill logic

Stage 2: public LOBSTER-style sample data

- Use message and orderbook splits where available.
- This is closer to the paper than FI-2010 because it has both orderbook snapshots and event messages.
- Current adapter expects standard LOBSTER columns:
  - message: seconds, event type, order ID, size, price, direction.
  - orderbook: ask price, ask size, bid price, bid size by level.
- LOBSTER prices are fixed-point by default and are converted with `price_scale=10000`.

Stage 3: optional FI-2010 pretraining

- Useful for benchmarking Attn-LOB mid-price classification.
- Less suitable for market-making replay because it is primarily a normalized mid-price forecasting benchmark.

## Implementation Order

1. Project hygiene:
   - Python 3.12.
   - uv-managed dependencies.
   - `src/mlfcs_gapa`.
   - `tests`.
   - typed config objects.

2. Data schema:
   - canonical orderbook snapshot dataframe.
   - canonical event/message dataframe.
   - loader for synthetic data.
   - loader adapter for LOBSTER-style data.

3. Simulator:
   - cash/inventory accounting.
   - quote generation.
   - fill model.
   - episode reset.
   - forced liquidation.
   - metrics.
   - unit tests for fills and PnL.

4. Baselines:
   - Fixed_1/2/3.
   - Random.
   - AS.
   - Inv-RL.
   - LOB-RL.

5. Attn-LOB:
   - PyTorch implementation of encoder.
   - pretraining dataset.
   - mid-price direction labels.
   - pretraining metrics table.

6. C-PPO:
   - Gymnasium environment wrapper.
   - Stable-Baselines3 PPO.
   - custom feature extractor using pretrained Attn-LOB.

7. D-DQN:
   - discrete-action Gymnasium environment.
   - custom PyTorch dueling Double DQN.
   - replay, target network, epsilon schedule.

8. Experiments:
   - overall table.
   - ablation table.
   - latency sweep.
   - runtime measurements.
   - attention visualization.
   - historical decision plot.

## Implementation Status

Completed first slice:

- Python package skeleton under `src/mlfcs_gapa`.
- Paper constants:
  - 10 LOB levels
  - 50-event window
  - 40 LOB features
  - horizon 10
  - alpha 1e-5
  - 100-share minimum trade unit
  - omega 10
  - 2000-event episodes
  - eta 0.5
  - zeta 0.01
  - max_bias 0.05
  - max_spread 0.1
- Canonical data schema:
  - orderbook snapshots
  - message/OSI aggregates
  - trade extrema summaries for historical fills
- Synthetic generator:
  - intentionally isolated in `mlfcs_gapa.data.synthetic`
  - emits canonical data only
  - includes intraday timestamps, 10-level depth, market/limit/withdraw aggregates, and trade extrema
- LOBSTER adapter:
  - `mlfcs-gapa convert-lobster`
  - converts standard LOBSTER message/orderbook CSV files to the canonical schema
  - maps event type 1 to limit orders, 2/3 to withdrawals, and 4/5 to executions
  - converts execution direction from resting limit-order side to aggressive market buy/sell volume
  - converts fixed-point prices with default scale `10000`
- LOB preprocessing:
  - relative-to-mid stationary price transform
  - z-normalized transformed price columns
  - max-normalized volume columns
- Mid-price direction labels:
  - output classes: down=0, stationary=1, up=2
  - invalid edge rows marked -1
- Attn-LOB encoder:
  - PyTorch implementation of Figure 1
  - Conv/Inception/attention output shape tests
  - returns attention weights for Figure 3
- Continuous action equations:
  - `delta = A1 * max_bias`
  - `reservation = mid - sign(inventory) * delta`
  - `spread = A2 * max_spread`
  - legal ask rounded up and bid rounded down to tick
- Hybrid reward components:
  - dampened PnL
  - trading PnL
  - inventory penalty
  - total reward as `DP + TP - IP`
- Historical replay primitives:
  - counterparty-price crossing fills
  - passive quote fills from historical trade extrema
  - equal-price probabilistic queue fills
  - cash/inventory/value accounting
  - forced liquidation at counterparty price
  - ND-PnL, PnLMAP, profit ratio, and inventory metrics
- Gymnasium C-PPO environment:
  - Dict observation with LOB state, 24-dimensional dynamic state, and 2-dimensional agent state
  - continuous `[0, 1]^2` action space
  - delayed decision state through `latency_events`
  - forced liquidation at episode end
- Gymnasium D-DQN environment:
  - same observation components as C-PPO
  - 8-action discrete space
  - actions 0-6 use fixed bid/ask tick offsets
  - action 7 closes inventory at counterparty price
- Non-learning baselines:
  - Fixed_1, Fixed_2, Fixed_3
  - Random level quoting over levels 1-5
  - Avellaneda-Stoikov
  - shared replay/fill/accounting/metrics path
- Tabular RL baselines:
  - Inv-RL with Lim-Gorse inventory/time state and 9 tick-offset actions
  - LOB-RL with Zhong `bidSpeed`, `askSpeed`, `avgmidChangeFrac`, `invSign`, `cumPnL`
  - Q-learning lookup table with replay/fill/accounting shared with other baselines
- Synthetic baseline command:
  - `mlfcs-gapa run-synthetic-baselines`
  - writes Fixed, Random, AS, Inv-RL, and LOB-RL metrics CSV plus trade log parquet
- Pretraining dataset builder:
  - normalized `(N, 50, 40)` LOB windows
  - horizon-10 direction labels
  - label alignment to the event at the end of each input window
- Pretraining loop:
  - trains any 3-class LOB classifier
  - reports macro precision, recall, F1, accuracy, loss, train N, validation N
  - synthetic Attn-LOB smoke command writes `attn_lob_pretrain_metrics.csv`
  - pretraining commands save model checkpoints for RL encoder initialization
- Table I supervised model classes:
  - FC-LOB
  - Conv-LOB
  - DeepLOB-style CNN-LSTM
  - Attn-LOB
  - all emit 3-class logits through the shared pretraining trainer
- C-PPO training command:
  - `mlfcs-gapa train-synthetic-ppo`
  - Stable-Baselines3 PPO with Attn-LOB feature extractor
  - writes model zip, metrics CSV, and trade log parquet
  - ablation switches:
    - `--lob-mode attn|mlp|none`
    - `--no-use-dynamic-state`
    - `--no-use-agent-state`
- D-DQN training command:
  - `mlfcs-gapa train-synthetic-ddqn`
  - custom PyTorch dueling Double DQN with Attn-LOB encoder
  - writes model checkpoint, metrics CSV, losses CSV, and trade log parquet
  - supports the same LOB/dynamic/agent state ablation switches
- Reporting and figure commands:
  - `mlfcs-gapa run-synthetic-latency-baselines`
  - writes `latency_metrics.csv`, `latency_trades.parquet`, and `latency_figure.png`
  - `mlfcs-gapa summarize-metrics` writes paper-scaled mean/std table columns
  - `mlfcs-gapa collect-metrics` concatenates per-job CSV metrics
  - `mlfcs-gapa plot-latency-figure` reproduces the Figure 2 panel structure
  - `mlfcs-gapa plot-decision-trace` reproduces the Figure 4 quote/mid/inventory trace structure
  - `mlfcs-gapa plot-synthetic-attention` reproduces the Figure 3 attention heatmap structure
  - `mlfcs-gapa benchmark-runtime-synthetic` writes Table III-style runtime rows
- Euler:
  - synced to `/cluster/scratch/piroth/mlfcs-gapa`
  - Python 3.12.8 venv installed
  - PyTorch 2.12.0+cu130 installed
  - CPU Slurm smoke job `67026477` passed on 2026-05-18:
    - 44 tests passed
    - synthetic generator wrote 300-event canonical data
    - baseline smoke wrote 7 methods: Fixed_1/2/3, Random, AS, Inv-RL, LOB-RL
    - Attn-LOB pretrain smoke wrote metrics and checkpoint
    - C-PPO smoke wrote model, metrics, and trade log
    - D-DQN smoke wrote model, metrics, losses, and trade log
    - elapsed `00:02:59`, max RSS about `1.2 GB`
  - Expanded CPU Slurm smoke job `25494` passed on 2026-05-19:
    - 51 tests passed
    - synthetic generator, seven baseline methods, pretrain checkpoint, C-PPO, D-DQN
    - latency metrics, summary metrics, latency figure, decision trace, and attention heatmap
    - elapsed `00:03:03`, max RSS about `1.1 GB`
- GPU job templates now request one GPU per task and use Slurm array caps of `%4`; do not submit GPU jobs outside those caps.
  - `scripts/euler/latency_agents_gpu_array.sh` trains/evaluates C-PPO and D-DQN across latencies with `--array=0-11%4`
  - `scripts/euler/ablation_agents_gpu_array.sh` runs Table IV variants with `--array=0-7%4`
  - `scripts/euler/runtime_cpu.sh` runs the Table III-style CPU runtime benchmark

Current test coverage:

- paper constants
- canonical synthetic data schema
- synthetic quote sanity
- LOB window generation
- mid-price labels
- Attn-LOB encoder/classifier shapes
- continuous action conversion
- hybrid reward equations
- replay matching/accounting/liquidation
- episode metrics
- pretraining window/label alignment
- dynamic state shape/range
- Gymnasium reset/step/terminal metrics
- discrete action mapping and D-DQN environment terminal metrics
- baseline quoting/evaluation
- tabular Inv-RL/LOB-RL state/action/training smoke
- classifier pretraining metrics
- pretraining model output shapes
- C-PPO Attn-LOB feature extractor and checkpoint loading
- D-DQN network and tiny training smoke
- C-PPO/D-DQN ablation feature modes:
  - no LOB state (`lob_mode=none`)
  - MLP over flattened LOB (`lob_mode=mlp`)
  - no dynamic state
- paper-scaled report aggregation
- latency, decision-trace, and attention-heatmap figure generation
- runtime benchmark command and metrics collection
- LOBSTER CSV adapter schema mapping

Paper-faithful interpretation choices made so far:

- Attention: Keras `MultiHeadAttention(num_heads=10, key_dim=16, output_shape=64)` maps 192-channel input to 160 attention channels and 64 output channels. The PyTorch implementation mirrors this with a 192-to-160 projection, 10-head attention, then 160-to-64 projection.
- Inventory penalty: `Inv_t` is interpreted as inventory in minimum-trade-unit lots. This is necessary for `zeta = 0.01` to be numerically meaningful with 100-share trade units. The implementation keeps this explicit in `inventory_penalty`. If strict raw-share inventory is required, it should be a named config switch because it changes reward scale drastically.
- LOB normalization: the paper cites stationary price normalization plus z-norm/max-norm, but does not give all details. The implementation uses relative-to-current-mid prices, then z-normalizes price columns inside each window and max-normalizes volume columns. This is closer to the paper text than the official demo code, which skips z-normalization.
- Fill model: if both sides fill in a single event, the primitive keeps one fill, matching the official demo's one-trade-result-per-step behavior. This case should be rare in real historical trade data but can happen in synthetic data.
- Agent state: implemented as the two factors stated in the paper, normalized inventory and episode progress. The official demo repeats these into 24 dimensions, but the paper does not state that repetition.
- Avellaneda-Stoikov defaults: `gamma=0.1`, `kappa=100.0` for a cent-tick stock around 16 currency units. `kappa=1.5`, a common toy default, produces a spread above 1 unit and almost never trades on the synthetic LOB. Final AS results should tune `gamma` and `kappa` on validation data and report the chosen values.
- Conv-LOB and DeepLOB: the paper names these baselines but does not fully specify every layer. The implemented Conv-LOB is a dilated fully convolutional network in the stated WaveNet spirit. The implemented DeepLOB uses the same LOB-level convolutional reduction and inception structure with LSTM temporal aggregation. These are marked as paper-faithful approximations pending any more precise architectural detail from the cited baseline implementations.
- Discrete D-DQN actions: the paper states 8 discrete actions, while the official demo metadata incorrectly reports 5 action values but implements action IDs 0-7. The replication implements the 8-action mapping: seven quote actions plus inventory-closing action 7.
- C-PPO/D-DQN pretraining: RL commands can load a saved Attn-LOB classifier checkpoint and extract encoder weights. Whether to freeze the encoder is configurable so the ablation plan can test both frozen and trainable transfer.
- Latency figure scaling: report helpers can display ND-PnL divided by `1e5` and Profit Ratio multiplied by `1e4`, matching the table scale annotations in the paper. Raw synthetic metrics are still preserved in CSV outputs.
- Ablation implementation: `w/o LOB state` maps to `lob_mode=none`; `w/o Attn-LOB` maps to `lob_mode=mlp`; `w/o Dynamic state` maps to `--no-use-dynamic-state`. The paper does not ablate agent state in Table IV, but the switch exists for sanity checks.

## Important Replication Risks

- Original data is unavailable, so reported absolute values cannot be matched.
- Official demo code is not a faithful implementation of every paper equation.
- Latency plot values are not tabulated.
- Random seeds and number of repeated runs are not specified in detail.
- Transaction costs are zero, which can overstate profitability.
- Fill assumptions are optimistic when using only L2 snapshots and not true queue position.
- Event-time and wall-clock-time features must be handled carefully; OSI/RV/RSI require timestamps, not just row offsets.

## Success Criteria

The replication is successful if it demonstrates the paper's methodological claims under substituted data:

- Attn-LOB pretraining reaches sensible directional classification performance.
- C-PPO outperforms Fixed, Random, AS, Inv-RL, and LOB-RL on inventory-adjusted metrics in at least synthetic and one public-data setting.
- Removing LOB state or Attn-LOB significantly hurts C-PPO.
- Latency hurts fixed/nonadaptive baselines more than trained RL agents.
- Attention and decision plots show plausible behavior: recent event focus, inventory-skewed quoting, and wider quotes in adverse/trending regimes.

## Euler Synthetic Run Log, 2026-05-19

Completed cluster validation:

- CPU smoke job `28271`: passed all 51 tests and all smoke commands.
- Pretraining array `28596`: completed FC-LOB, Conv-LOB, DeepLOB, and Attn-LOB synthetic pretraining. The synthetic directional labels are weak; these numbers validate the pipeline, not the paper's Table I.
- Runtime job `30992`: completed and wrote `runs/runtime-cpu/30992/runtime_metrics.csv`.
- Main agent array `30994`: completed C-PPO and D-DQN with 20,000 training timesteps.
- Latency agent array `30996`: completed C-PPO and D-DQN for latencies `[1, 5, 10, 20, 50, 100]` under the four-GPU array cap.
- Ablation array `30999`: completed all eight Table IV-style variants under the four-GPU array cap.
- CPU latency baseline job `35862`: completed Fixed, Random, AS, Inv-RL, and LOB-RL for the same latency grid.

Current consolidated Euler artifacts:

- `runs/combined_metrics_30994_30996_30999_35862.csv`
- `runs/latency_metrics_30996_35862.csv`
- `runs/ablation_summary_30999.csv`
- `runs/summary_metrics_30994_30996_30999_35862.csv`
- `runs/latency_figure_30996_35862.png`

Important diagnostic outcome:

- C-PPO collapsed to a no-trade/max-spread policy in the main latency-1 run:
  - 1,951 replay rows
  - 0 fills
  - mean raw action-implied spread about `0.10`, the paper `max_spread`
  - terminal PnL, ND-PnL, PnLMAP, inventory, and profit ratio all equal to 0
- The C-PPO Table IV ablations also produced no-trade rows, so the current synthetic C-PPO result is not a valid reproduction of the paper's qualitative claim.
- D-DQN traded and produced non-zero metrics, but the ablation ordering is not paper-like on the current synthetic run:
  - full D-DQN: positive PnL in ablation run
  - w/o dynamic state: also positive and larger in this seed
  - w/o LOB state: negative
- The synthetic full-latency plot renders structurally, but the current curves are diagnostic outputs, not paper-faithful final results.

Next calibration work before scaling seeds:

1. Keep the paper equations and fill/accounting path unchanged.
2. Re-run C-PPO with explicit paper-aligned PPO learning rate `1e-4`; earlier Euler scripts relied on the SB3 default `3e-4`.
3. Use the newly exposed PPO knobs only as hyperparameters, not methodology changes:
   - `PPO_LEARNING_RATE`
   - `PPO_GAMMA`
   - `PPO_GAE_LAMBDA`
   - `PPO_CLIP_RANGE`
   - `PPO_ENT_COEF`
   - `PPO_VF_COEF`
   - `PPO_MAX_GRAD_NORM`
4. If C-PPO still converges to no-trade, tune the synthetic generator rather than the paper reward:
   - ensure passive quotes have a realistic positive spread-capture edge under low latency;
   - reduce excessive adverse selection in synthetic market-order/trade-extrema generation;
   - preserve LOB schema and keep synthetic logic separate from the paper simulator.
5. Only after C-PPO trades plausibly should we run multi-seed tables and polish the paper figures.

Follow-up calibration result:

- Four 60,000-timestep C-PPO calibration jobs (`37107`, `37111`, `37114`, `37118`) varied learning rate and entropy coefficient while keeping the direct `[0, 1]` action space. All completed successfully, and all still converged to the identical max-spread/no-fill policy.
- A local policy-surface check showed the synthetic environment does contain positive-reward fixed continuous policies, for example paper action approximately `[bias=0.5, spread=0.3]`.
- The likely issue is action parameterization for PPO, not only data: SB3's Gaussian policy starts naturally around zero, which is an edge of the paper `[0, 1]` action range. The official Tensorforce metadata also exposes `[-1, 1]` actions even though the paper equations are written for `[0, 1]`.
- The implementation now supports normalized PPO actions: external PPO actions are in `[-1, 1]`, then mapped internally to the paper action `A = (a + 1) / 2` before applying the paper quote equations. This keeps the paper action equations intact while making initial PPO actions correspond to the center of the paper action range.
- A local 512-timestep normalized-action smoke run produced non-zero fills, confirming that the no-trade pathology is addressed mechanically before launching the next Euler calibration.
- Euler normalized-action jobs `38766` and `38797` still converged to max-spread/no-fill with SB3's default Gaussian exploration scale.
- Euler narrow-exploration jobs:
  - `40846`, `policy_log_std_init=-1`: 8 fills, PnL about `-1.0`.
  - `40849`, `policy_log_std_init=-2`: 28 fills, PnL about `8.0`, mean absolute inventory about `5.33`, mean quoted spread about `0.055`.
- This confirms the immediate C-PPO blocker is PPO exploration/action parameterization. `policy_log_std_init=-2` is the current best synthetic C-PPO calibration point, but it is still a calibration result, not a final paper table run.
- Before launching more C-PPO jobs, check `squeue -u "$USER"` because unrelated Euler GPU jobs may already be running under the account. On 2026-05-19 after `40849`, three non-MLFCS GPU jobs were active, so no further MLFCS GPU jobs were launched.
