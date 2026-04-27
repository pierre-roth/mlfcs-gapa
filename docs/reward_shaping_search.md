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

## Results

No new reward-search results yet. Fill this section from `ppo_episodes.csv`,
`dqn_episodes.csv`, and `paper_baseline_episodes.csv` as the Euler jobs finish.

Minimum comparison fields:

| run_name | policy | pnl | reward | avg_abs_position | avg_spread | fill_rate | turnover | positive episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

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
