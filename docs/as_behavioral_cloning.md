# AS Behavioral Cloning Warm Starts

This log tracks experiments that initialize PPO/DQN from an Avellaneda-Stoikov
teacher before RL fine-tuning. The goal is to test whether a good market-making
prior prevents PPO/DQN from learning high-turnover or high-inventory behavior.

## Implementation

- `BC_AS_INIT=true` enables an AS behavioral-cloning warm start inside
  `train_ppo` or `train_dqn`.
- PPO cloning maps the AS quote to the continuous action vector for the active
  `CONTINUOUS_ACTION_MODE` and trains the PPO actor mean with MSE.
- DQN cloning maps the AS quote to the nearest discrete action and trains the
  DQN Q-head with cross entropy.
- `BC_AS_FREEZE_BACKBONE=true` freezes the Attn-LOB/fusion backbone during
  cloning and trains only the final policy head. The full model is unfrozen
  again for RL fine-tuning.
- `BC_AS_EPOCHS`, `BC_AS_MAX_SAMPLES_PER_DAY`, and `BC_AS_LOSS_WEIGHT` control
  cloning cost and strength.

## Initial Cases

The first BC batch uses the inventory-aware reward setting from the reward
search (`inv_lot1`) because the prior DQN diagnostics showed that explicit
inventory feedback made DQN profitable and low-inventory, while paper-reward DQN
did not learn to exit breached inventory.

Settings:

- `TRADE_UNIT_OVERRIDE=1`
- `REWARD_MODE=hybrid`
- `REWARD_USE_DAMPENED_PNL=false`
- `REWARD_USE_TRADING_PNL=false`
- `REWARD_USE_INVENTORY_PENALTY=true`
- `REWARD_ZETA=0.000005`
- `REWARD_SPREAD_PENALTY_SCALE=0`
- `BC_AS_INIT=true`
- `BC_AS_EPOCHS=4`
- `BC_AS_FREEZE_BACKBONE=true`
- `BC_AS_MAX_SAMPLES_PER_DAY=12000`

## Submitted Runs

| group | algo | dataset | symbol | run_name | pretrain | train | eval | baseline | status | result |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `bc_as_inv_lot1` | PPO | synthetic | 000858 | `piroth2_bc_as_inv_lot1_ppo_synth_000858_20260427_175423` | 64931585 | 64931587 | 64931590 | 64931593 | active | pending |
| `bc_as_inv_lot1` | DQN | synthetic | 000858 | `piroth2_bc_as_inv_lot1_dqn_synth_000858_20260427_175423` | 64931596 | 64931599 | 64931604 | 64931606 | active | pending |
| `bc_as_inv_lot1` | PPO | real | AAPL | `piroth2_bc_as_inv_lot1_ppo_real_AAPL_20260427_175423` | 64931610 | 64931615 | 64931620 | 64931623 | active | pending |
| `bc_as_inv_lot1` | PPO | real | GOOGL | `piroth2_bc_as_inv_lot1_ppo_real_GOOGL_20260427_175423` | 64931627 | 64931631 | 64931634 | 64931636 | active | pending |
| `bc_as_inv_lot1` | DQN | real | AAPL | `piroth2_bc_as_inv_lot1_dqn_real_AAPL_20260427_175423` | 64931640 | 64931643 | 64931647 | 64931649 | active | pending |

## Results

No new behavioral-cloning results yet. Fill this section from
`as_bc_ppo_history.csv`, `as_bc_dqn_history.csv`, `ppo_episodes.csv`,
`dqn_episodes.csv`, and matching baseline CSVs.

Minimum comparison fields:

| run_name | algo | bc_loss_or_acc | pnl | reward | avg_abs_position | avg_spread | fill_rate | turnover | positive episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Interpretation Plan

- Compare BC+RL against the matching non-BC reward-search run.
- If BC improves early training but not held-out PnL, inspect whether it is
  washed out during RL and consider freezing the backbone longer or lowering the
  RL learning rate.
- If BC improves held-out PnL and inventory control, test whether the gain
  persists under a more paper-faithful reward and under both synthetic and real
  data.
