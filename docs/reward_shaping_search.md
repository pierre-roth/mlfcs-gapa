# Reward Shaping Search

This log tracks experiments that search over linear combinations of the reward
terms exposed by the authors' commented hybrid reward. The purpose is diagnostic:
the final paper-faithful reference remains the executable author reward, but the
previous PPO/DQN evidence shows that reward alignment is the strongest blocker.

## Implementation

- `REWARD_MODE=hybrid` now supports explicit linear weights:
  `REWARD_PNL_WEIGHT`, `REWARD_TRADING_PNL_WEIGHT`,
  `REWARD_INVENTORY_PENALTY_WEIGHT`, and `REWARD_SPREAD_PENALTY_WEIGHT`.
- `TRADE_UNIT_OVERRIDE` can reduce the quote/order unit from the paper lot size
  of 100 shares to 1 share. Inventory normalization, inventory guards, AS/fixed
  baselines, PPO, DQN, and DQN diagnostics use the effective trade unit.
- PPO entropy now supports a linear schedule from `PPO_ENTROPY_COEF` to
  `PPO_ENTROPY_COEF_FINAL`, written into `c_ppo_history.csv`.
- The initial search is a bounded space-filling screen. Bayesian optimization
  should use these results as the first observations before submitting the next
  candidate batch.

## Initial Search Space

All initial reward-search cases use `TRADE_UNIT_OVERRIDE=1`.

| label | pnl term | trading pnl term | inventory penalty | spread penalty |
| --- | ---: | ---: | ---: | ---: |
| `pnl_lot1` | `1.0 * pnl` | off | off | off |
| `inv_lot1` | `1.0 * pnl` | off | `zeta=0.000005` | off |
| `trdinv_lot1` | `1.0 * pnl` | `0.25 * matched_pnl` | `zeta=0.000005` | off |

PPO uses a larger schedule than the earlier bounded probes:
`PPO_EPOCHS=24`, `PPO_ROLLOUTS_PER_EPOCH=128`, `PPO_UPDATE_EPOCHS=6`,
`TORCH_BATCH_SIZE=2048`, `PPO_INITIAL_LOG_STD=-1.4`, and entropy decays from
`0.006` to `0.0002`.

DQN uses the previously better bounded tuning cadence:
`TORCH_EPOCHS=10`, `DQN_REPLAY_SIZE=250000`, `DQN_MIN_REPLAY=4096`,
`DQN_UPDATE_INTERVAL=96`, `DQN_TARGET_UPDATE_STEPS=1000`, and epsilon decays
from `0.50` to `0.05`.

## Submitted Runs

| group | algo | dataset | symbol | run_name | pretrain | train | eval | baseline | status | result |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `rew_pnl_lot1` | PPO | synthetic | 000858 | `piroth2_rew_pnl_lot1_ppo_synth_000858_20260427_175423` | 64931400 | 64931403 | 64931405 | 64931407 | active | pending |
| `rew_pnl_lot1` | DQN | synthetic | 000858 | `piroth2_rew_pnl_lot1_dqn_synth_000858_20260427_175423` | 64931409 | 64931411 | 64931415 | 64931417 | active | pending |
| `rew_pnl_lot1` | PPO | real | AAPL | `piroth2_rew_pnl_lot1_ppo_real_AAPL_20260427_175423` | 64931419 | 64931421 | 64931427 | 64931432 | active | pending |
| `rew_pnl_lot1` | PPO | real | GOOGL | `piroth2_rew_pnl_lot1_ppo_real_GOOGL_20260427_175423` | 64931435 | 64931437 | 64931441 | 64931444 | active | pending |
| `rew_inv_lot1` | PPO | synthetic | 000858 | `piroth2_rew_inv_lot1_ppo_synth_000858_20260427_175423` | 64931446 | 64931448 | 64931450 | 64931452 | active | pending |
| `rew_inv_lot1` | DQN | synthetic | 000858 | `piroth2_rew_inv_lot1_dqn_synth_000858_20260427_175423` | 64931454 | 64931457 | 64931459 | 64931461 | active | pending |
| `rew_inv_lot1` | PPO | real | AAPL | `piroth2_rew_inv_lot1_ppo_real_AAPL_20260427_175423` | 64931463 | 64931466 | 64931468 | 64931478 | active | pending |
| `rew_inv_lot1` | PPO | real | GOOGL | `piroth2_rew_inv_lot1_ppo_real_GOOGL_20260427_175423` | 64931481 | 64931484 | 64931486 | 64931488 | active | pending |
| `rew_trdinv_lot1` | PPO | synthetic | 000858 | `piroth2_rew_trdinv_lot1_ppo_synth_000858_20260427_175423` | 64931491 | 64931494 | 64931497 | 64931503 | active | pending |
| `rew_trdinv_lot1` | DQN | synthetic | 000858 | `piroth2_rew_trdinv_lot1_dqn_synth_000858_20260427_175423` | 64931506 | 64931512 | 64931515 | 64931521 | active | pending |
| `rew_trdinv_lot1` | PPO | real | AAPL | `piroth2_rew_trdinv_lot1_ppo_real_AAPL_20260427_175423` | 64931524 | 64931527 | 64931532 | 64931535 | active | pending |
| `rew_trdinv_lot1` | PPO | real | GOOGL | `piroth2_rew_trdinv_lot1_ppo_real_GOOGL_20260427_175423` | 64931538 | 64931541 | 64931543 | 64931548 | active | pending |
| `rew_inv_lot1` | DQN | real | AAPL | `piroth2_rew_inv_lot1_dqn_real_AAPL_20260427_175423` | 64931551 | 64931555 | 64931560 | 64931563 | active | pending |
| `rew_inv_lot1` | DQN | real | GOOGL | `piroth2_rew_inv_lot1_dqn_real_GOOGL_20260427_175423` | 64931565 | 64931572 | 64931575 | 64931583 | active | pending |
| `bo2_zeta2` | PPO | synthetic | 000858 | `piroth2_bo2_zeta2_ppo_synth_000858_20260427_225530` | 64962399 | 64962401 | 64962403 | 64962406 | active | pending |
| `bo2_zeta10` | PPO | synthetic | 000858 | `piroth2_bo2_zeta10_ppo_synth_000858_20260427_225530` | 64962410 | 64962412 | 64962414 | 64962416 | active | pending |
| `bo2_trd010` | PPO | synthetic | 000858 | `piroth2_bo2_trd010_ppo_synth_000858_20260427_225530` | 64962421 | 64962423 | 64962425 | 64962427 | active | pending |
| `xsym_pnl_lot1` | PPO | synthetic | 000001 | `piroth2_xsym_pnl_lot1_ppo_synth_000001_20260427_225530` | 64962430 | 64962432 | 64962434 | 64962438 | active | pending |
| `xsym_pnl_lot1` | PPO | synthetic | 002415 | `piroth2_xsym_pnl_lot1_ppo_synth_002415_20260427_225530` | 64962453 | 64962458 | 64962460 | 64962463 | active | pending |
| `real250_inv_lot1` | DQN | real | AAPL | `piroth2_real250_inv_lot1_dqn_real_AAPL_20260427_225530` | 64962505 | 64962507 | 64962510 | 64962511 | active | pending |
| `real250_inv_lot1` | DQN | real | GOOGL | `piroth2_real250_inv_lot1_dqn_real_GOOGL_20260427_225530` | 64962519 | 64962521 | 64962524 | 64962527 | active | pending |
| `real250_pnl_lot1` | PPO | real | AAPL | `piroth2_real250_pnl_lot1_ppo_real_AAPL_20260427_225530` | 64962512 | 64962514 | 64962515 | 64962517 | active | pending |
| `real250_pnl_lot1` | PPO | real | GOOGL | `piroth2_real250_pnl_lot1_ppo_real_GOOGL_20260427_225530` | 64962529 | 64962530 | 64962532 | 64962533 | active | pending |

## Results

First batch completed. All runs used `TRADE_UNIT_OVERRIDE=1`, so absolute PnL
is not directly comparable to the earlier 100-share-lot experiments without
accounting for trade size. The qualitative result is still clear: PPO becomes
low-inventory and profitable on synthetic data, while DQN remains weak on
synthetic data but works surprisingly well on the short real-data splits.

| run_name | policy | pnl | reward | avg_abs_position | avg_spread | fill_rate | turnover | positive episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rew_pnl_lot1_ppo_synth_000858` | PPO | +0.4995 | +0.4995 | 0.2051 | 0.0340 | 0.0635 | 8.43e3 | 59/60 |
| `rew_inv_lot1_ppo_synth_000858` | PPO | +0.4688 | +0.4680 | 0.1691 | 0.0341 | 0.0624 | 8.28e3 | 59/60 |
| `rew_trdinv_lot1_ppo_synth_000858` | PPO | +0.5153 | +0.6587 | 0.2083 | 0.0343 | 0.0635 | 8.42e3 | 58/60 |
| matching synthetic 000858 | AS | +0.5363 | varies by reward | 3.9454 | 0.0279 | 0.1189 | 1.62e4 | 37/60 |
| `rew_pnl_lot1_dqn_synth_000858` | DQN | -1.0735 | -1.0735 | 8.0240 | 0.0219 | 0.0564 | 8.61e3 | 8/60 |
| `rew_inv_lot1_dqn_synth_000858` | DQN | -0.7020 | -0.9807 | 6.4880 | 0.0256 | 0.0759 | 1.16e4 | 14/60 |
| `rew_trdinv_lot1_dqn_synth_000858` | DQN | -0.4553 | -0.5572 | 5.9528 | 0.0228 | 0.0901 | 1.50e4 | 21/60 |
| `rew_pnl_lot1_ppo_real_AAPL` | PPO | -0.1300 | -0.1300 | 0.6073 | 0.0281 | 0.6585 | 7.25e3 | 2/4 |
| `rew_inv_lot1_ppo_real_AAPL` | PPO | -0.3875 | -0.3876 | 0.4969 | 0.0342 | 0.5860 | 6.36e3 | 1/4 |
| `rew_trdinv_lot1_ppo_real_AAPL` | PPO | +0.1325 | +0.0583 | 0.7238 | 0.0364 | 0.4999 | 5.34e3 | 3/4 |
| AAPL real baseline | Fixed_1 | +0.0900 | +0.0884 | 2.8405 | 0.0245 | 0.5989 | 5.44e3 | 1/3 |
| AAPL real baseline | AS | 0.0000 | 0.0000 | 0.0000 | 0.2448 | 0.0000 | 0.00 | 0/3 |
| `rew_pnl_lot1_ppo_real_GOOGL` | PPO | +0.5829 | +0.5829 | 0.5870 | 0.0353 | 0.4547 | 4.24e3 | 5/7 |
| `rew_inv_lot1_ppo_real_GOOGL` | PPO | +0.2957 | +0.2956 | 0.3799 | 0.0360 | 0.4930 | 4.94e3 | 5/7 |
| `rew_trdinv_lot1_ppo_real_GOOGL` | PPO | -0.0800 | -0.1404 | 0.2892 | 0.0342 | 0.5023 | 5.11e3 | 2/7 |
| GOOGL real baseline | Fixed_1 | +2.2129 | +2.2121 | 1.9752 | 0.0215 | 0.4993 | 3.99e3 | 5/7 |
| GOOGL real baseline | AS | 0.0000 | 0.0000 | 0.0000 | 0.2445 | 0.0000 | 0.00 | 0/7 |
| `rew_inv_lot1_dqn_real_AAPL` | DQN | +3.7650 | +3.7610 | 3.1862 | 0.0432 | 0.4741 | 5.72e3 | 3/4 |
| `rew_inv_lot1_dqn_real_GOOGL` | DQN | +3.2129 | +3.2121 | 1.4054 | 0.0373 | 0.4529 | 4.76e3 | 6/7 |

Interpretation:

- The trade-unit change is important. With unit size 1, synthetic PPO no longer
  exhibits the high-inventory/high-turnover failure mode and reaches roughly AS
  PnL on 000858.
- The best first-batch synthetic PPO result is still only marginally above or
  below AS depending on the exact reward; it is not a decisive replication win.
- Synthetic DQN remains poor even with unit size 1. Adding the trading-PnL term
  helps relative to pure PnL, but all three DQN synthetic variants are negative.
- On real data, DQN with the inventory penalty beats AS and fixed baselines on
  the short AAPL/GOOGL splits. This needs a larger split because the first real
  batch has only 4 AAPL PPO/DQN eval episodes and 7 GOOGL eval episodes.
- Real PPO is mixed: GOOGL pure PnL is positive but below Fixed_1, while AAPL
  only becomes positive in the trading-PnL plus inventory-penalty setting.

## Follow-Up Results

The follow-up batch completed with all Slurm jobs exiting `0:0`. It tested a
small local reward-search around the first synthetic PPO result, cross-symbol
synthetic checks, and larger real-data splits with `REAL_EVENT_STRIDE=250`.

| run_name | policy | pnl | reward | avg_abs_position | avg_spread | fill_rate | turnover | positive episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bo2_zeta2_ppo_synth_000858` | PPO | +0.5028 | +0.5024 | 0.1996 | 0.0343 | 0.0635 | 8.42e3 | 59/60 |
| `bo2_zeta10_ppo_synth_000858` | PPO | +0.4843 | +0.4820 | 0.2297 | 0.0367 | 0.0601 | 7.95e3 | 58/60 |
| `bo2_trd010_ppo_synth_000858` | PPO | +0.4785 | +0.5309 | 0.1754 | 0.0340 | 0.0639 | 8.48e3 | 58/60 |
| matching synthetic 000858 | AS | +0.5363 | varies by reward | 3.9454 | 0.0279 | 0.1189 | 1.62e4 | 37/60 |
| `xsym_pnl_lot1_ppo_synth_000001` | PPO | +0.1073 | +0.1073 | 0.1118 | 0.0331 | 0.0172 | 2.28e2 | 32/40 |
| matching synthetic 000001 | AS | -0.0402 | -0.0402 | 2.3337 | 0.0294 | 0.0242 | 3.59e2 | 17/40 |
| `xsym_pnl_lot1_ppo_synth_002415` | PPO | +0.0693 | +0.0693 | 0.0732 | 0.0321 | 0.0193 | 6.54e2 | 26/40 |
| matching synthetic 002415 | AS | -0.4462 | -0.4462 | 2.8150 | 0.0290 | 0.0239 | 9.31e2 | 9/40 |
| `real250_inv_lot1_dqn_real_AAPL` | DQN | +0.2700 | +0.2697 | 1.4108 | 0.0438 | 0.4733 | 5.03e3 | 6/10 |
| AAPL real250 baseline | Random | +0.1889 | +0.1887 | 1.1090 | 0.0709 | 0.2684 | 2.32e3 | 6/9 |
| AAPL real250 baseline | AS | 0.0000 | 0.0000 | 0.0000 | 0.2154 | 0.0000 | 0.00 | 0/9 |
| `real250_pnl_lot1_ppo_real_AAPL` | PPO | -0.4770 | -0.4770 | 0.5000 | 0.0346 | 0.5579 | 6.05e3 | 2/10 |
| `real250_inv_lot1_dqn_real_GOOGL` | DQN | +0.1656 | +0.1654 | 1.3599 | 0.0356 | 0.4654 | 4.14e3 | 8/16 |
| GOOGL real250 baseline | Fixed_1 | +0.0240 | +0.0238 | 1.6797 | 0.0223 | 0.5043 | 3.56e3 | 6/15 |
| GOOGL real250 baseline | AS | 0.0000 | 0.0000 | 0.0000 | 0.2359 | 0.0000 | 0.00 | 0/15 |
| `real250_pnl_lot1_ppo_real_GOOGL` | PPO | -0.0350 | -0.0350 | 0.4096 | 0.0199 | 0.6739 | 5.80e3 | 6/16 |

Follow-up interpretation:

- The local search did not beat the best first-batch synthetic PPO result or AS
  on 000858. `zeta=0.000002` is the best follow-up 000858 candidate, but the
  difference from pure PnL is small.
- Pure-PnL PPO with trade unit 1 generalizes well to synthetic stress symbols:
  it beats AS on 000001 and 002415 with much lower inventory and turnover.
- Real-data DQN remains the strongest real-data RL result in this branch. On the
  `REAL_EVENT_STRIDE=250` split it beats AS and Fixed_1 on GOOGL, and beats AS,
  Fixed_1, and Random on AAPL by a small margin.
- Real-data PPO is still weak at stride 250. It is low-inventory but does not
  overcome adverse selection/high fill rates on these short splits.

## Bayesian Optimization Round

The next batch stops AS behavioral cloning and treats the completed reward
screen as seed observations for a conservative batch-BO round. The acquisition
logic is practical rather than fully automated inside the training loop: propose
near the best observed region, include a few uncertainty probes, and evaluate
the same candidates with PPO and DQN under longer training budgets.

Search dimensions:

- `TRADE_UNIT_OVERRIDE`: `1` near the current optimum, plus a unit-2 probe.
- `REWARD_TRADING_PNL_WEIGHT`: `0.0`, `0.05`, `0.15`, `0.30`.
- `REWARD_ZETA`: `0.000001`, `0.000002`, `0.000003`, `0.000005`,
  `0.000008`.
- `REWARD_INVENTORY_PENALTY_WEIGHT`: `1.0`, plus one weaker `0.3` probe.
- `REWARD_SPREAD_PENALTY_SCALE`: mostly `0`, plus one tiny `0.005` probe.
- `MAKER_REBATE_PER_SHARE`: `0`, `0.0015`, and `0.0025`. The nonzero values
  are intended to be realistic for Nasdaq-listed AAPL/GOOGL: current Nasdaq
  displayed add-liquidity credits for stocks priced at or above $1 are on the
  order of one to a few mils per share. The rebate is optional and defaults to
  zero.
- Real-data calibration sweeps `REAL_EVENT_STRIDE=100,250,500` for AAPL and
  GOOGL. Earlier results showed that stride materially changes fill density and
  adverse selection, so it is treated as a data-calibration axis rather than a
  reward parameter.

Training budget:

- PPO: `PPO_EPOCHS=36`, `PPO_ROLLOUTS_PER_EPOCH=160`,
  `PPO_UPDATE_EPOCHS=8`, entropy scheduled from `0.008` to `0.0001`.
- DQN: `TORCH_EPOCHS=16`, `DQN_REPLAY_SIZE=350000`,
  `DQN_UPDATE_INTERVAL=96`, `DQN_TARGET_UPDATE_STEPS=1000`, epsilon scheduled
  from `0.55` to `0.04`.
- Encoder pretraining is shared per dataset/symbol through an explicit
  checkpoint pass-through, avoiding redundant pretrains for each reward
  candidate.

Planned candidate labels:

| label | trade unit | trading pnl weight | zeta | inventory penalty weight | spread penalty scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bo3_z1_trd0_u1` | 1 | 0.00 | 0.000001 | 1.0 | 0 |
| `bo3_z3_trd0_u1` | 1 | 0.00 | 0.000003 | 1.0 | 0 |
| `bo3_z2_trd05_u1` | 1 | 0.05 | 0.000002 | 1.0 | 0 |
| `bo3_z2_trd15_u1` | 1 | 0.15 | 0.000002 | 1.0 | 0 |
| `bo3_z8_trd30_u1` | 1 | 0.30 | 0.000008 | 1.0 | 0 |
| `bo3_z5_i03_u1` | 1 | 0.00 | 0.000005 | 0.3 | 0 |
| `bo3_z2_sp0005_u1` | 1 | 0.00 | 0.000002 | 1.0 | 0.005 |
| `bo3_z2_trd0_u2` | 2 | 0.00 | 0.000002 | 1.0 | 0 |
| `bo3_z2_trd0_r15_u1` | 1 | 0.00 | 0.000002 | 1.0 | 0 |
| `bo3_z2_trd05_r25_u1` | 1 | 0.05 | 0.000002 | 1.0 | 0 |

Synthetic 000858 runs all ten candidates with both PPO and DQN. Real AAPL and
GOOGL run six candidates (`bo3_z1_trd0_u1`, `bo3_z2_trd05_u1`,
`bo3_z8_trd30_u1`, `bo3_z2_trd0_u2`, `bo3_z2_trd0_r15_u1`, and
`bo3_z2_trd05_r25_u1`) with both PPO and DQN at strides 100, 250, and 500.
The `r15` and `r25` suffixes mean maker rebates of `$0.0015/share` and
`$0.0025/share`.

### Bayesian Optimization Results

The BO batch `20260428_103736` completed with all checked Slurm jobs exiting
`0:0`.

Synthetic 000858:

- Best PPO: `bo3_z2_trd0_u2`, PnL `+1.1287`, reward `+1.1283`,
  positive episodes `58/60`, median PnL `+0.9900`, minimum PnL `-0.0200`,
  avg abs position `0.3991`. Same-run AS PnL was `+1.0727`, so this is the
  first non-BC synthetic PPO setting that cleanly edges AS on the main symbol.
- Best DQN: `bo3_z2_trd0_r15_u1`, PnL `+0.4313`, reward `+0.3999`,
  positive episodes `49/60`, median PnL `+0.3632`, avg abs position `2.6632`.
  Same-run AS PnL was `+0.7101`, so DQN is positive but still not competitive
  with AS on synthetic 000858.

Real-data aggregate over AAPL/GOOGL and strides 100/250/500:

| algo | candidate | mean PnL over six cells | worst cell mean PnL | mean positive rate | worst positive rate | mean avg abs position |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DQN | `bo3_z1_trd0_u1` | +1.3804 | -0.3285 | 0.5762 | 0.4000 | 1.5134 |
| DQN | `bo3_z2_trd0_u2` | +1.2605 | -0.3290 | 0.4836 | 0.3000 | 2.7574 |
| DQN | `bo3_z2_trd05_r25_u1` | +0.6490 | -0.5687 | 0.5095 | 0.3500 | 1.3102 |
| PPO | `bo3_z2_trd0_u2` | -0.0295 | -0.7140 | 0.3857 | 0.2000 | 1.0255 |

Real-data interpretation:

- DQN is the only real-data learner with positive aggregate PnL, but no real
  candidate is consistently positive across both symbols and all strides.
- Stride 500 produces the largest PnL, but has few held-out episodes; stride
  100/250 are more useful for consistency.
- PPO remains unreliable on real data under this reward/stride search.

### Consistency Confirmation

The next batch is deliberately narrow. It stops broad search and tests whether
the best settings are repeatably positive.

- Synthetic: PPO `bo3_z2_trd0_u2` across symbols `000858`, `000001`, `002415`
  and seeds `7`, `11`, `17`, with a larger 24-day split and `PPO_EPOCHS=40`.
- Real: DQN candidates `bo3_z1_trd0_u1` and `bo3_z2_trd0_u2` across AAPL/GOOGL,
  strides 100/250, and seeds `7`, `11`, using an 18-day split and
  `TORCH_EPOCHS=20`.
- Consistency bar: mean PnL > 0, median PnL > 0, positive episode rate at least
  65%, and no large inventory-driven tail loss.

## Decision Rule

- Treat a setting as promising if held-out PnL is positive, average absolute
  inventory is materially below the paper-reward DQN failure mode, and it beats
  the comparable AS/fixed/random baselines on the same split.
- Treat a setting as replication-relevant only if the result is clearly labeled
  as non-paper-faithful when it changes reward weights or `TRADE_UNIT_OVERRIDE`.
- Use the best three completed observations to seed a Bayesian-optimization
  batch over `reward_trading_pnl_weight`, `reward_zeta`,
  `reward_inventory_penalty_weight`, `reward_spread_penalty_weight`, and
  `trade_unit_override`.
- The second batch is a small BO-style/local search around the first-batch
  optimum: `zeta=0.000002`, `zeta=0.000010`, and
  `trading_pnl_weight=0.10` on synthetic 000858, plus cross-symbol and larger
  real-data confirmations.
