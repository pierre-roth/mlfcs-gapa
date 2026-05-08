# Paper Alignment Notes

This branch targets a paper-faithful replication of *Market Making with Deep
Reinforcement Learning from Limit Order Books* on synthetic LOB data. The paper
authors' reference code in
`/Users/piroth/Downloads/Market-Making-with-Deep-Reinforcement-Learning-from-Limit-Order-Books-master`
is useful as an implementation hint, but it is incomplete and internally
inconsistent, so this branch treats it as intent rather than an executable spec.

## Architecture Mapping

- LOB input: `T x 40 x 1`, with `T=50` by default.
- Pretraining-comparison encoders:
  - `fclob`: fully connected baseline with `2000 -> 1024 -> 256 -> 64`
    hidden representation.
  - `convlob`: fully convolutional baseline with temporal dilated
    convolutions and a 64-dimensional pooled representation.
  - `deeplob`: DeepLOB-style CNN/inception front-end with LSTM temporal
    aggregation.
  - `attnlob`: paper/reference-code attention architecture used as the default
    trading encoder.
- Early convolutions:
  - `Conv2D(32, 1x2, stride 1x2)`
  - two `Conv2D(32, 4x1, same)` layers
  - `Conv2D(32, 1x5, stride 1x5)`
  - two `Conv2D(32, 4x1, same)` layers
  - `Conv2D(32, 1x4)`
  - two `Conv2D(32, 4x1, same)` layers
- Inception block:
  - `1x1 -> 3x1`, 64 filters
  - `1x1 -> 5x1`, 64 filters
  - `3x1 max-pool -> 1x1`, 64 filters
  - concatenate to 192 channels.
- Attention block:
  - the authors use Keras `MultiHeadAttention(num_heads=10, key_dim=16, output_shape=64)`.
  - this branch implements the same dimensions manually in PyTorch because
    PyTorch `nn.MultiheadAttention` requires `embed_dim` to be divisible by
    `num_heads`, while the paper/code combination uses 192 input channels and
    10 heads.
- Pretraining head:
  - 3 classes: up, stationary, down.
- Trading state:
  - LOB encoder output: 64
  - dynamic market state: 24
  - agent state: 24
  - fused to a 64-dimensional hidden state before PPO/DQN heads.
- LOB feature order now follows the authors' CSV/model path: all ask levels
  first (`ask1_price, ask1_volume, ..., ask10_volume`) followed by all bid
  levels. Earlier branch revisions interleaved ask/bid by level, which is not
  what the reference `ask.csv + bid.csv` concatenation produces.
- Paper-prose price normalization is available as `lob_price_z_norm=True`.
  This first applies the reference-code stationary price transform
  (`price / contemporaneous mid - 1`) and then z-normalizes each price column
  inside the 50-event LOB window. Volume columns remain per-window
  max-normalized. The released code contains the z-normalization helpers but
  comments out the active lines, so this option is for paper-description table
  replication rather than executable-code ablations.

The implementation lives in [piroth/models.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/models.py).

## Reference-Code Discrepancies

- `network/network.py` matches the screenshot and is the strongest source for
  the Attn-LOB encoder shape.
- `environment/env_continuous.py` comments describe multiple continuous action
  conventions. The executable path uses a two-dimensional action for price bias
  and spread. This branch uses that two-dimensional C-PPO action surface.
- `environment/env_discrete.py` declares `num_values=5`, but `action2order`
  contains actions `0..7`. This branch exposes 8 D-DQN actions to match the
  executable action mapping.
- The same discrete action path also suppresses the ask side when inventory is
  below `-10 * TRADE_UNIT`, and suppresses the bid side when inventory is above
  `+10 * TRADE_UNIT`. The first bounded DQN pilot missed this guard, causing
  very large average absolute inventory. `DiscreteActionPolicy` now matches the
  guard and has a focused regression test.
- `network/network.py:get_model()` contains what appears to be a bug: when
  `with_market_state=True`, it appends `agent_state` to the dense input instead
  of `market_state`. This may explain why our `no_dynamic` ablation did not
  degrade PPO. The branch keeps the corrected market-state fusion by default,
  and exposes the literal reference-code behavior as
  `author_market_state_alias=True` for controlled ablations.
- The executable continuous reward in `environment/env_continuous.py` is not the
  hybrid reward suggested by nearby comments. It is `pnl - spread_punishment`;
  the matched-PnL, dampened-PnL, and inventory-punishment terms are commented
  out. This branch now defaults to that author-executable reward via
  `reward_mode="author_pnl"`, while retaining `reward_mode="hybrid"` for
  ablations.
- The authors' `match()` function has one `trade_price, trade_volume` output.
  If both bid and ask would fill in one step, the later buy branch overwrites
  the earlier sell branch. This branch now defaults to the same one-net-fill
  behavior via `matching_mode="author_single"`, while retaining
  `matching_mode="multi_fill"` for diagnostics.
- The executable continuous action maps `action[0]` to a 0.05 price adjustment
  and `action[1]` to a 0.1 absolute spread scale, but the same environment
  declares the action space as `[-1, 1]` while the executable comment says
  actions are in `[0, 1]`. This branch defaults to
  `continuous_action_mode="author"`, which maps PPO's tanh output from
  `[-1, 1]` into that intended `[0, 1]` scale before applying the author
  formula. The literal unshifted behavior remains available as
  `continuous_action_mode="author_raw"`, and the previous bounded-spread
  transform remains available as `continuous_action_mode="bounded"`.

## Synthetic-Data Contract

Synthetic data must be meaningful at the paper's 2000-event episode scale. The
first-order quality gate is not model performance; it is whether the generated
LOB has enough movement, spread variation, persistent order flow, depth
imbalance variation, and plausible fill curves for market-making policies to be
nontrivial.

The visual report is therefore part of the replication surface, not just a
debug artifact. It shows:

- daily midprice and latent fair value
- 2000-event episode gallery
- spread and return distributions
- rolling 2000-event signed order flow
- depth heatmap and snapshots
- fill probability by quote distance
- baseline evaluation tables when available
- synthetic quality score and flags

The report is produced by [piroth/visualizer.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/visualizer.py).

## Current Empirical Finding

The strongest discrepancy is now the executable reward, not the model
architecture. On the larger 000858 PPO check, pure-PnL C-PPO becomes profitable
with more data and compute (`+58.46` held-out PnL), but AS still wins (`+90.77`).
With the executable author spread penalty restored, C-PPO instead overtrades
and loses heavily (`-164.33` PnL), while shaped reward improves by tightening
spreads.

The cross-symbol stress runs on 000001 and 002415 confirm the same mechanism.
Under the author reward, C-PPO evaluates at `-34.32` and `-47.33` PnL with
roughly 36-40% fill rates and 0.011 spreads. Under pure PnL, the same PPO
configuration evaluates at `+5.04` and `+7.54` PnL with wider 0.032 spreads and
much lower turnover, beating AS on these two stress symbols where AS also loses
money. This supports treating reward specification and synthetic market
calibration as the next major paper-replication risks before interpreting DQN
results.

The first bounded DQN pilot is also negative. On a reduced 000858 split, author
DQN evaluates at `-82.03` PnL and pure-PnL DQN at `-90.72`, while AS earns
`+58.94` under the same data split. Both DQN variants carry very large average
absolute inventory (`622` and `828`) and high turnover. A concrete
paper-alignment discrepancy was then found in the discrete inventory-limit
guard described above. The fixed policy passed the focused Euler test, but the
matched reduced DQN rerun still failed: author DQN evaluated at `-90.00` PnL
with `828.55` average absolute inventory, and pure-PnL DQN evaluated at
`-88.16` PnL with `830.97` average absolute inventory. The guard prevents
adding to an already breached side, but it does not force liquidation. The
current DQN blocker is now behavioral and algorithmic, not just this single
paper-alignment bug.

The action/inventory diagnostic confirms this. In the fixed author DQN replay,
inventory is already at or beyond the guard threshold on 62.6% of steps, but the
liquidation action is selected on 0.0% of those breached steps. The fixed
pure-PnL DQN shows the same pattern: 63.5% breached steps and 0.0% breached-step
liquidation. The liquidation Q-value is also far below the selected action
(`-6.79` mean Q margin in the author run, `-6.53` in pure PnL), so the network
has learned to prefer continuing ordinary quote actions while stuck near the
inventory boundary.

A forced-liquidation counterfactual improves PnL materially but does not close
the gap. Forcing action 7 whenever inventory is already breached moves author
DQN from `-90.00` to `-30.50` PnL and pure-PnL DQN from `-88.16` to `-32.78`
PnL, while cutting average absolute inventory by about `404-407` shares. This
shows that failure to exit breached inventory explains a large part of the DQN
loss. It is not the whole story: the counterfactual still trails AS (`+58.94`
PnL), so DQN also needs better entry/quote selection or a reward signal that
teaches inventory risk earlier than terminal liquidation.

The inventory-penalty diagnostic confirms the direction. With a non-author
hybrid reward that removes the spread penalty and applies explicit inventory
risk (`reward_zeta=0.05`), DQN evaluates at `+36.94` PnL with `94.39` average
absolute inventory on the same reduced 000858 split. That is still below AS
(`+58.94`) and should not be reported as the paper-faithful DQN result, but it
shows the implementation can learn a profitable low-inventory DQN policy when
the reward gives earlier inventory feedback.

The action diagnostic for that checkpoint shows no breached-inventory steps on
held-out replay, a 9.4% overall liquidation-action rate, and a much smaller
liquidation Q-margin (`-0.41`) than the paper-reward DQN. This is strong
evidence that the architecture and DQN mechanics are not intrinsically broken;
the remaining paper-faithful gap is the reward/action credit assignment under
the authors' executable objective.

The larger 000858 confirmation strengthens that conclusion. With the same
inventory-penalty diagnostic reward on 16 days and 60k events/day, DQN reaches
`+52.13` held-out PnL, `+33.16` shaped reward, and only `33.59` average absolute
inventory across 60 test episodes; every held-out episode is positive. This
nearly matches serious pure-PnL C-PPO (`+58.46`) but still trails AS (`+90.77`).
It is useful evidence about implementation capacity, not a substitute for a
paper-faithful D-DQN result.

Cross-symbol inventory-penalty confirmations are also positive. On 000001,
DQN reaches `+19.23` held-out PnL with `24.53` average absolute inventory,
beating pure-PnL C-PPO (`+5.04`) and AS (`-9.22`). On 002415, DQN reaches
`+24.08` held-out PnL with `18.10` average absolute inventory, again beating
pure-PnL C-PPO (`+7.54`) and AS (`-23.76`). All 80 cross-symbol held-out
episodes are positive. This makes the reward specification the clearest
remaining blocker to a paper-faithful D-DQN result.

A bounded paper-faithful DQN tuning run reinforces that conclusion. Keeping the
executable author reward (`author_pnl`, spread penalty scale 100) but increasing
exploration and slowing the update cadence produced worse held-out behavior on
000858: `-120.43` PnL, `-131.12` reward, and `627.81` average absolute inventory,
with only 4 of 60 held-out episodes positive. Training reward improved across
epochs while training PnL fell to `-87.99`, matching the earlier pattern that the
author reward can be optimized by behavior that is economically poor. The action
diagnostic confirms the same non-liquidating breached-inventory failure mode:
40.1% of replayed steps begin beyond the paper inventory guard, the liquidation
rate on those breached steps is 0.0%, and the liquidation Q-value is on average
`7.21` below the selected action under breach. The policy uses liquidation
frequently overall (`21.9%` of steps), but primarily when it is not already
breached, so the issue is state-conditional value preference rather than the
absence of liquidation from the action space.

## Cluster Workflow

The `mm-drl-lob` branch provided the Euler job structure: stage-specific sbatch
files, a shared common shell helper, and a submission wrapper with per-stage
resource overrides. This branch mirrors that shape for the synthetic-paper
pipeline in [cluster/submit_piroth2.sh](/Users/piroth/Documents/projects/mlfcs-gapa/cluster/submit_piroth2.sh).

All substantive runs should happen on Euler under the `ls_math` account.

## Table II Tabular Baselines

The paper Table II includes two learned baselines that are absent from the
released reference repository: Inventory-RL and LOB-RL. This branch implements
paper-aligned tabular versions in
[piroth/tabular_baselines.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/tabular_baselines.py)
so the real-data Table II can include the missing rows.

Inventory-RL uses a tabular Q-table over `(inventory_bucket, remaining_time_bucket)`.
Inventory buckets are flat plus small/medium/large long and short imbalance;
the time axis is divided into 12 episode-progress buckets. The action space is
the nine bid/ask quote-offset pairs from `{0, 1, 2}` ticks, using the same trade
unit, inventory guard, matching, terminal liquidation, and metrics as the
C-PPO/D-DQN environment.

LOB-RL uses a tabular Q-table over the Zhong/Bergstrom/Ward-style aggregated
state: bid-side pressure flag, ask-side pressure flag, mid-price-change bucket,
inventory bucket, and cumulative-PnL bucket. Its actions are quote neither,
ask only, bid only, or both at the touch. Strong long inventory restricts the
admissible actions to selling or doing nothing, and strong short inventory
restricts them to buying or doing nothing.

Both baselines train with epsilon-greedy Q-learning and evaluate greedily.
They intentionally reuse the same real-data split and paper-style hybrid reward
settings as the current Table II run. The exact original paper implementation
details for these baselines are not fully specified, so the output should be
reported as transparent, paper-aligned tabular baselines rather than claimed as
byte-for-byte reproductions of the authors' private experiment code.

### Real-Data Table II Run, 2026-05-08

The first end-to-end real-data Table-II-style run completed on AAPL and GOOGL
with raw real events, 2,000-event windows, the paper reward settings, the
paper-style z-normalized LOB features, and the implemented Table II agents:
AS, Fixed_1/2/3, Random, Inventory-RL, LOB-RL, C-PPO, and D-DQN. Full numeric
results are stored in
[docs/results/real_table2_20260508.csv](/Users/piroth/Documents/projects/mlfcs-gapa/docs/results/real_table2_20260508.csv).

The AAPL test set has no convincing learned-policy replication success. Fixed_3
is roughly flat and best by mean PnL (`+0.0625`). LOB-RL is also near flat
(`-0.0417`), followed by Fixed_2 (`-0.7917`) and Inventory-RL (`-1.4583`).
The neural policies lose more: C-PPO evaluates at `-4.8750` PnL and D-DQN at
`-8.0000`. AS is also negative (`-7.3542`) because on these raw real-event
windows its quotes fill much more often than in the earlier strided smoke
setting.

The GOOGL test set is similarly disappointing for neural RL. Random is best by
mean PnL (`+1.8125`), LOB-RL is near flat (`+0.0417`), and Fixed_3 loses only
`-0.5833`. Inventory-RL loses `-1.7708`. D-DQN (`-12.9375`) and C-PPO
(`-17.7917`) both underperform simple baselines, and AS is worst at `-19.8958`.

The completed table supports the negative replication finding the project is
trying to make explicit: under the current paper-faithful real-data setup,
neural RL does not dominate simple quoting or tabular baselines. LOB-RL is the
most robust learned row in this run, but it is only near-flat rather than a
strong market-making result.

The matching real-data encoder pretraining comparison also completed. Under the
paper-style z-normalized real-data setup, DeepLOB is the strongest encoder on
both symbols: AAPL final eval accuracy is `0.7238` versus AttnLOB `0.6932`,
FC-LOB `0.6879`, and ConvLOB `0.6766`; GOOGL final eval accuracy is `0.6490`
versus AttnLOB `0.5975`, FC-LOB `0.5960`, and ConvLOB `0.5870`. This differs
from a literal expectation that AttnLOB should necessarily be best, but it is
internally consistent with the weak downstream neural-RL results on these real
NASDAQ feeds.

## Real NASDAQ Data

The paper-style pipeline now supports `data_source=real` via
[piroth/real_data.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/real_data.py).
The loader consumes the existing Euler layout:

```text
/cluster/work/math/piroth/mlfcs-gapa/data/processed/{AAPL,GOOGL}/YYYYMMDD/
  ask.csv bid.csv price.csv trades.csv msg.csv
```

The real files are much denser than the synthetic paper-scale data: at the
NASDAQ open, 2,000 raw events cover only about one second for these feeds. The
loader therefore exposes `real_event_stride` and `events_per_day_override` so
bounded experiments can use strided event streams while preserving the paper
`SyntheticDay` interface. Initial real-data experiments use
`real_event_stride=100`, `events_per_day_override=60000`, six train days, and
three test days for AAPL and GOOGL under both author-spread-penalty and pure-PnL
reward settings.

Active real-data run set:

```text
piroth2_real_author_AAPL_20260427_105334: pretrain=64880911, PPO=64880913, eval=64880916, baseline=64880918
piroth2_real_purepnl_AAPL_20260427_105334: pretrain=64880919, PPO=64880922, eval=64880924, baseline=64880926
piroth2_real_author_GOOGL_20260427_105334: pretrain=64880929, PPO=64880931, eval=64880933, baseline=64880936
piroth2_real_purepnl_GOOGL_20260427_105334: pretrain=64880938, PPO=64880940, eval=64880942, baseline=64880945
```

Initial real-data baseline results show that `REAL_EVENT_STRIDE=100` is a
working compatibility setting but not yet a clean paper-scale sampling choice.
AAPL quality score is `63.85` and GOOGL quality score is `64.27`; both are
flagged for high trade density, overly directional 2000-event windows, and
little order-flow persistence. On AAPL, AS has `0.00` PnL and no fills; under
the author reward it still receives `-382.38` reward because the spread penalty
fires while flat. Fixed_1 loses `-30.67` PnL, Fixed_2 loses `-1.63`, Fixed_3
loses `-2.33`, and Random loses `-27.29`. On GOOGL, AS is near flat at `-0.33`
PnL, while Fixed_1/2/3 and Random lose `-9.04`, `-4.29`, `-13.04`, and
`-22.29` PnL. These are useful smoke baselines, but final real-data
interpretation needs better event sampling or episode construction before the
real runs can be treated as paper-scale evidence.

The first completed real PPO evals are the two AAPL runs. Author-reward C-PPO
evaluates at `-18.29` PnL, `-44.33` reward, `40.09` average absolute inventory,
`0.0229` average spread, `0.696` fill rate, and `5.23e5` turnover, with 9 of 24
episodes positive. Pure-PnL C-PPO is similar: `-17.08` PnL/reward, `39.20`
average absolute inventory, `0.0248` average spread, `0.691` fill rate, and
`5.19e5` turnover.

The GOOGL evals tell the same story with a small pure-PnL improvement.
Author-reward C-PPO evaluates at `-14.21` PnL, `-41.38` reward, `44.75` average
absolute inventory, `0.0253` average spread, `0.742` fill rate, and `5.25e5`
turnover, with 11 of 24 episodes positive. Pure-PnL C-PPO evaluates at `-7.38`
PnL/reward, `44.74` average absolute inventory, `0.0350` average spread, `0.596`
fill rate, and `4.17e5` turnover, with 10 of 24 episodes positive. This is not
a replication success: PPO beats the worst active baselines, but still trails AS
and the wider fixed policies on PnL. The high fill rates also reinforce that
current real-event sampling is not yet calibrated to a clean paper-scale
market-making task.

## Synthetic Generator Variants

The simulator now has opt-in knobs for two real-data-inspired effects while
preserving previous defaults:

- `order_flow_memory`: repeats the prior noise-taker direction with a configured
  probability, increasing market-order sign persistence.
- `volatility_cluster_strength` and `volatility_cluster_persistence`: make
  fair-value noise temporarily larger after shocks.

The first bounded generator sweep keeps the author reward fixed and changes the
market structure:

```text
piroth2_synth_flowvol_author_000858_20260427_105410: pretrain=64881054, PPO=64881056, eval=64881058, baseline=64881060
  ORDER_FLOW_MEMORY=0.35, VOLATILITY_CLUSTER_STRENGTH=0.45, VOLATILITY_CLUSTER_PERSISTENCE=0.992
  quality score=99.63, flags=0
  AS pnl=+94.22/reward=-153.34
  C-PPO eval pnl=-191.50/reward=-339.70/avg_abs_position=30.02/avg_spread=0.0128/fill_rate=0.597/turnover=7.89e6

piroth2_synth_noinform_author_000858_20260427_105410: pretrain=64881063, PPO=64881065, eval=64881067, baseline=64881069
  INFORMED_MARKET_ORDER_PROB=0.0, NOISE_MARKET_ORDER_PROB=0.48
  quality score=71.77, flag=median 2000-event window is too flat
  AS pnl=+2.72/reward=-2112.00
  C-PPO eval pnl=-0.97/reward=-166.90/avg_abs_position=13.53/avg_spread=0.0114/fill_rate=0.271/turnover=3.57e6
```

The result is diagnostic rather than a fix. The `flowvol` generator is the
stronger synthetic-data direction because it restores order-flow persistence and
volatility clustering without failing the quality gate, but it makes the
author-reward PPO failure more severe: C-PPO quotes very tightly, fills on about
60% of steps, and loses `-191.50` PnL while AS earns `+94.22`. The `noinform`
variant confirms the opposite failure mode: removing informed flow makes the
market too flat, so PPO damage becomes small but the experiment no longer tests
a meaningful market-making task. This supports the current conclusion that
synthetic quality matters, but the executable author spread penalty remains a
dominant source of the replication gap even on a high-quality synthetic variant.
