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
- `BC_AS_FREEZE_BACKBONE=true` freezes the Attn-LOB encoder by default during
  cloning and trains the fusion layer plus final policy head. Set
  `BC_AS_FREEZE_ENCODER_ONLY=false` to reproduce the first-batch behavior, which
  froze the whole backbone including the randomly initialized fusion layer. The
  full model is unfrozen again for RL fine-tuning.
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
| `xsym_bc2_inv_lot1` | PPO | synthetic | 000001 | `piroth2_xsym_bc2_inv_lot1_ppo_synth_000001_20260427_225530` | 64962444 | 64962446 | 64962449 | 64962452 | active | pending |
| `xsym_bc2_inv_lot1` | PPO | synthetic | 002415 | `piroth2_xsym_bc2_inv_lot1_ppo_synth_002415_20260427_225530` | 64962465 | 64962468 | 64962499 | 64962502 | active | pending |
| `bc2_inv_lot1` | PPO | synthetic | 000858 | `piroth2_bc2_inv_lot1_ppo_synth_000858_20260427_225530` | 64962535 | 64962537 | 64962539 | 64962541 | active | pending |
| `bc2_inv_lot1` | DQN | synthetic | 000858 | `piroth2_bc2_inv_lot1_dqn_synth_000858_20260427_225530` | 64962543 | 64962548 | 64962575 | 64962577 | active | pending |
| `bc2_inv_lot1` | DQN | real | AAPL | `piroth2_bc2_inv_lot1_dqn_real_AAPL_20260427_225530` | 64962580 | 64962584 | 64962586 | 64962589 | active | pending |
| `bc2_inv_lot1` | DQN | real | GOOGL | `piroth2_bc2_inv_lot1_dqn_real_GOOGL_20260427_225530` | 64962591 | 64962594 | 64962596 | 64962598 | active | pending |

## Results

First batch completed. These runs used the older overly restrictive cloning
freeze, so they are useful diagnostics but not the final BC answer.

| run_name | algo | bc_loss_or_acc | pnl | reward | avg_abs_position | avg_spread | fill_rate | turnover | positive episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bc_as_inv_lot1_ppo_synth_000858` | PPO | loss 0.1611 | +0.5632 | +0.5619 | 0.2600 | 0.0383 | 0.0587 | 7.76e3 | 57/60 |
| `bc_as_inv_lot1_dqn_synth_000858` | DQN | acc 0.5777 | -0.6280 | -0.8703 | 5.4158 | 0.0236 | 0.1100 | 1.83e4 | 16/60 |
| `bc_as_inv_lot1_ppo_real_AAPL` | PPO | loss 1.7656 | -0.2450 | -0.2451 | 0.5008 | 0.0340 | 0.5860 | 6.36e3 | 1/4 |
| `bc_as_inv_lot1_ppo_real_GOOGL` | PPO | loss 1.7709 | +0.0029 | +0.0028 | 0.3244 | 0.0240 | 0.5726 | 5.72e3 | 2/7 |
| `bc_as_inv_lot1_dqn_real_AAPL` | DQN | acc 0.0000 | -0.8250 | -0.8256 | 1.0123 | 0.0275 | 0.4348 | 4.96e3 | 1/4 |

Interpretation:

- BC helped PPO on synthetic 000858 slightly: `+0.5632` versus `+0.4688` for
  the matching non-BC inventory-penalty PPO and `+0.5363` for AS.
- BC did not help DQN in this first form. DQN synthetic stayed negative, and
  DQN real AAPL became worse than the non-BC DQN.
- The DQN real AAPL BC classifier accuracy of `0.0000` and the high real-data
  PPO BC losses indicate the first implementation froze too much randomly
  initialized network capacity. The code now freezes only the encoder by
  default and leaves the fusion layer plus final policy head trainable.

## Follow-Up Results

The follow-up batch completed with the corrected cloning freeze: by default only
the Attn-LOB encoder is frozen during cloning, while the fusion layer and final
policy/Q head remain trainable. All Slurm jobs exited `0:0`.

| run_name | algo | bc_loss_or_acc | pnl | reward | avg_abs_position | avg_spread | fill_rate | turnover | positive episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bc2_inv_lot1_ppo_synth_000858` | PPO | loss 0.0070 | +0.0552 | -0.0204 | 4.7771 | 0.0299 | 0.0549 | 8.00e3 | 33/60 |
| matching non-BC `rew_inv_lot1_ppo_synth_000858` | PPO | n/a | +0.4688 | +0.4680 | 0.1691 | 0.0341 | 0.0624 | 8.28e3 | 59/60 |
| matching synthetic 000858 | AS | n/a | +0.5363 | varies by reward | 3.9454 | 0.0279 | 0.1189 | 1.62e4 | 37/60 |
| `bc2_inv_lot1_dqn_synth_000858` | DQN | loss 0.8133 | -1.2135 | -1.3290 | 6.4631 | 0.0280 | 0.0377 | 7.22e3 | 5/60 |
| matching non-BC `rew_inv_lot1_dqn_synth_000858` | DQN | n/a | -0.7020 | -0.9807 | 6.4880 | 0.0256 | 0.0759 | 1.16e4 | 14/60 |
| `xsym_bc2_inv_lot1_ppo_synth_000001` | PPO | loss 0.0076 | +0.0273 | +0.0207 | 0.9474 | 0.0446 | 0.0047 | 7.30e1 | 15/40 |
| matching non-BC `xsym_pnl_lot1_ppo_synth_000001` | PPO | n/a | +0.1073 | +0.1073 | 0.1118 | 0.0331 | 0.0172 | 2.28e2 | 32/40 |
| `xsym_bc2_inv_lot1_ppo_synth_002415` | PPO | loss 0.0100 | -0.1813 | -0.1979 | 1.5931 | 0.0398 | 0.0087 | 3.56e2 | 13/40 |
| matching non-BC `xsym_pnl_lot1_ppo_synth_002415` | PPO | n/a | +0.0693 | +0.0693 | 0.0732 | 0.0321 | 0.0193 | 6.54e2 | 26/40 |
| `bc2_inv_lot1_dqn_real_AAPL` | DQN | loss 1.8749 | -0.1360 | -0.1365 | 1.3974 | 0.0451 | 0.4563 | 5.44e3 | 4/10 |
| matching non-BC `real250_inv_lot1_dqn_real_AAPL` | DQN | n/a | +0.2700 | +0.2697 | 1.4108 | 0.0438 | 0.4733 | 5.03e3 | 6/10 |
| `bc2_inv_lot1_dqn_real_GOOGL` | DQN | loss 1.8335 | +0.1431 | +0.1423 | 1.6999 | 0.0442 | 0.3486 | 3.41e3 | 8/16 |
| matching non-BC `real250_inv_lot1_dqn_real_GOOGL` | DQN | n/a | +0.1656 | +0.1654 | 1.3599 | 0.0356 | 0.4654 | 4.14e3 | 8/16 |

Follow-up interpretation:

- The corrected freeze did not make AS behavioral cloning a reliable
  improvement. It worsened PPO on synthetic 000858 and both synthetic
  cross-symbol checks relative to the non-BC reward-search runs.
- BC also worsened synthetic DQN and real AAPL DQN. Real GOOGL DQN remained
  positive but was slightly below the non-BC DQN.
- The first-batch PPO synthetic 000858 BC result remains the only BC run that
  beat its matching non-BC run, but that result did not reproduce under the
  better freeze policy. Current evidence does not justify scaling BC before
  fixing reward/data calibration.

## Interpretation Plan

- Compare BC+RL against the matching non-BC reward-search run.
- If BC improves early training but not held-out PnL, inspect whether it is
  washed out during RL and consider freezing the backbone longer or lowering the
  RL learning rate.
- If BC improves held-out PnL and inventory control, test whether the gain
  persists under a more paper-faithful reward and under both synthetic and real
  data.
