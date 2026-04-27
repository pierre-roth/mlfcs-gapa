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
