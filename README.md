# Synthetic Attn-LOB Market Making Replication

This branch is a paper-faithful replication track for *Market Making with Deep Reinforcement Learning from Limit Order Books*, using synthetic LOB data instead of the original Shenzhen data.

Current scope:
- on-the-fly synthetic event generation
- explicit FIFO order book
- agent-based market dynamics
- export in the paper-compatible `ask/bid/price/trades/msg` format
- richer simulator internals in `event_log.csv` and `latent.csv`
- paper-style dynamic state, agent state, LOB normalization, and Attn-LOB pretraining labels
- paper-style random, fixed, and Avellaneda-Stoikov evaluations
- PyTorch implementations of Attn-LOB, C-PPO, and D-DQN
- synthetic data visual report

Main files:
- [paper/paper.tex](/Users/piroth/Documents/projects/mlfcs-gapa/paper/paper.tex)
- [docs/piroth2_simulator.md](/Users/piroth/Documents/projects/mlfcs-gapa/docs/piroth2_simulator.md)
- [docs/paper_alignment.md](/Users/piroth/Documents/projects/mlfcs-gapa/docs/paper_alignment.md)
- [piroth/config.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/config.py)
- [piroth/simulator.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/simulator.py)
- [piroth/paper_features.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/paper_features.py)
- [piroth/paper_env.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/paper_env.py)
- [piroth/models.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/models.py)
- [piroth/training.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/training.py)
- [piroth/visualizer.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/visualizer.py)
- [piroth/baselines.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/baselines.py)
- [piroth/results_summary.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/results_summary.py)
- [piroth/diagnostics.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/diagnostics.py)
- [cluster/submit_piroth2_ppo_sweep.sh](/Users/piroth/Documents/projects/mlfcs-gapa/cluster/submit_piroth2_ppo_sweep.sh)

Cluster jobs:

```bash
# recommended dependency-chained pipeline
RUN_NAME=piroth2_main SYMBOL=000001 MODE=medium cluster/submit_piroth2.sh pipeline

# synthetic data diagnostics, paper baselines, visual report
RUN_NAME=piroth2_diag SYMBOL=000001 MODE=medium cluster/submit_piroth2.sh diagnostics

# paper baseline table only
KIND=paper-baselines cluster/submit_piroth2.sh evaluate

# multi-seed synthetic data quality validation
VALIDATION_SYMBOLS=000001,000858,002415 VALIDATION_SEEDS=7,11,17,23 cluster/submit_piroth2.sh validate-data

# latency and ablation experiments
KIND=latency-suite cluster/submit_piroth2.sh evaluate
KIND=ablation-suite cluster/submit_piroth2.sh suite

# end-to-end paper suite: data, baselines, pretrain, PPO, DQN, evaluation, visualizer
cluster/submit_piroth2.sh suite

# visual report only
cluster/submit_piroth2.sh report

# Attn-LOB pretraining
cluster/submit_piroth2.sh pretrain

# C-PPO / D-DQN
cluster/submit_piroth2.sh train-ppo
cluster/submit_piroth2.sh train-dqn

# trained policy evaluation
KIND=evaluate-ppo cluster/submit_piroth2.sh evaluate
KIND=evaluate-dqn cluster/submit_piroth2.sh evaluate

# PPO robustness sweep without rerunning baselines/DQN
STAMP=20260424_232000 SYMBOLS="000001 000858 002415" SEEDS="11 17" cluster/submit_piroth2_ppo_sweep.sh
```

Per-stage resources can be overridden without editing sbatch files, for example:

```bash
TRAIN_PPO_TIME=12:00:00 TRAIN_PPO_GPUS=1 cluster/submit_piroth2.sh train-ppo
```

This branch does not pre-generate the full dataset. Days are generated deterministically on demand and can optionally be exported for inspection.

The first artifact to inspect after a diagnostics or visualize job is:

```text
${OUTPUT_ROOT}/${RUN_NAME}/visual_report/index.html
```

The first artifact to inspect after `validate-data` is:

```text
${OUTPUT_ROOT}/${RUN_NAME}/index.html
```

Latest smoke quality gate on Euler:

```text
RUN_NAME=piroth2_validation_final_20260424_181659
cases=12
pass_rate=100%
score_mean=98.71
score_min=93.78
flags_total=0
```

Latest focused cluster tests:

```text
JOB=64744928
tests=22 passed
scope=author alignment: simulator, baselines, paper env, models, features, results summary
JOB=64745745
tests=13 passed
scope=action scaling, author market-state alias, paper env, LOB feature order
JOB=64748108
tests=14 passed
scope=configurable PPO initial log-std/spread bias, paper env, LOB feature order
warning=PyTorch padding='same' warning for even convolution kernels
```

Author-code alignment status:

```text
Reference path: /Users/piroth/Downloads/Market-Making-with-Deep-Reinforcement-Learning-from-Limit-Order-Books-master
Implemented defaults after comparison:
- LOB tensor order follows ask.csv then bid.csv, not interleaved ask/bid levels.
- Reward defaults to the executable author path: pnl - flat-inventory wide-spread penalty.
- Matching defaults to the author one-net-fill behavior.
- C-PPO action scaling maps tanh outputs from [-1, 1] into the authors' intended [0, 1] action2order scale.
- The authors' market-state fusion bug is available as author_market_state_alias=True for ablation, but the default uses the intended dynamic state.

Important: pre-20260425 neural result tables are not directly comparable to current author-aligned runs.
```

Latest author-aligned smoke:

```text
RUN_NAME=piroth2_author_actionfix_smoke_000001_20260425_113320
jobs=64745686 -> 64745687 -> 64745688, baseline=64745689
synthetic_quality_score=95.78, flags=0
AS baseline: pnl=-0.50, avg_abs_position=92.69, avg_spread=0.0293, fill_rate=0.0276
C-PPO eval: pnl=-1.00, avg_abs_position=3.85, avg_spread=0.0284, fill_rate=0.0770
```

Latest author-aligned medium comparison:

```text
RUN_NAME pattern: piroth2_author_actionfix_medium_<symbol>_20260425_113629
symbols=000001,000858,002415
settings: MODE=medium, EVENTS_PER_DAY_OVERRIDE=30000, EPISODE_LENGTH=2000, TORCH_BATCH_SIZE=1024
quality scores: 000001=99.60, 000858=99.51, 002415=99.19, flags=0 for all
000001: C-PPO pnl=-2.50, AS pnl=-7.92, C-PPO avg_abs_position=3.49, C-PPO avg_spread=0.0287
000858: C-PPO pnl=-5.83, AS pnl=2.25, C-PPO avg_abs_position=13.53, C-PPO avg_spread=0.0284
002415: C-PPO pnl=-6.67, AS pnl=1.58, C-PPO avg_abs_position=5.89, C-PPO avg_spread=0.0281
interpretation: action-fixed PPO is stable and low-inventory, but still quotes above the author's 0.02 spread-penalty threshold.
```

Latest tight PPO author-aligned sweep:

```text
RUN_NAME pattern: piroth2_author_tightppo_<symbol>_20260425_122827
settings: MODE=full with NUM_DAYS=8/TRAIN_DAYS=4/TEST_DAYS=4, EVENTS_PER_DAY_OVERRIDE=30000,
          PPO_EPOCHS=8, PPO_UPDATE_EPOCHS=4, PPO_ENTROPY_COEF=0.001,
          PPO_INITIAL_LOG_STD=-2.0, PPO_INITIAL_SPREAD_BIAS=-1.1
000001 jobs: 64748124 -> 64748125 -> 64748126, baseline=64748127, report=64748128
000858 jobs: 64748129 -> 64748130 -> 64748131, baseline=64748132, report=64748133
002415 jobs: 64748134 -> 64748135 -> 64748136, baseline=64748137, report=64748138
000001 result: C-PPO pnl=-41.67, reward=-189.00, avg_spread=0.0127, fill_rate=0.377; AS pnl=-7.92
000858 result: C-PPO pnl=-109.17, reward=-416.33, avg_spread=0.0132, fill_rate=0.484; AS pnl=2.25
002415 result: C-PPO pnl=-64.17, reward=-290.00, avg_spread=0.0132, fill_rate=0.418; AS pnl=1.58
interpretation: this setting lowers author reward penalty but crosses into overtrading/adverse-selection losses.
```

Latest balanced PPO probe:

```text
RUN_NAME pattern: piroth2_author_balancedppo_000858_b<spreadbias>_20260425_124559
purpose: locate the spread-bias region between stable-too-wide (-0.70) and overtrading-too-tight (-1.10)
settings: SYMBOL=000858, PPO_INITIAL_LOG_STD=-1.8, PPO_ENTROPY_COEF=0.003, PPO_EPOCHS=6
bias -0.85 jobs: 64749260 -> 64749263 -> 64749265
bias -0.95 jobs: 64749270 -> 64749275 -> 64749278
bias -0.85 result: C-PPO pnl=-5.83, reward=-2522.75, avg_spread=0.0284, fill_rate=0.135
bias -0.95 result: C-PPO pnl=-109.17, reward=-416.33, avg_spread=0.0132, fill_rate=0.484
next probe: biases -0.88/-0.90/-0.92, same 000858 setup
bias -0.88 jobs: 64750585 -> 64750587 -> 64750589
bias -0.90 jobs: 64750593 -> 64750594 -> 64750595
bias -0.92 jobs: 64750596 -> 64750597 -> 64750598
bias -0.88 result: C-PPO pnl=-80.17, reward=-711.42, avg_spread=0.0154, fill_rate=0.422
bias -0.90 result: C-PPO pnl=-40.42, reward=-1056.33, avg_spread=0.0181, fill_rate=0.337
bias -0.92 result: C-PPO pnl=-109.17, reward=-416.33, avg_spread=0.0132, fill_rate=0.484
interpretation: author reward is strongly misaligned with PnL here; lower reward penalty tracks tighter spreads and higher turnover, not better market-making PnL.
```

Latest hybrid-reward probe:

```text
RUN_NAME pattern: piroth2_hybridreward_000858_b<spreadbias>_20260425_132022
purpose: test the reward terms commented in the authors' env_continuous.py against the executable author_pnl default
settings: SYMBOL=000858, REWARD_MODE=hybrid, PPO_INITIAL_LOG_STD=-1.8, PPO_ENTROPY_COEF=0.003, PPO_EPOCHS=6
bias -0.70 jobs: 64751612 -> 64751613 -> 64751614
bias -0.85 jobs: 64751615 -> 64751616 -> 64751617
bias -0.70 result: C-PPO pnl=-5.83, reward=-2560.68, avg_spread=0.0284, fill_rate=0.135
bias -0.85 result: C-PPO pnl=-16.67, reward=-2581.73, avg_spread=0.0282, fill_rate=0.149
interpretation: the commented hybrid terms do not fix the reward issue under this setup; spread penalty still dominates and deterministic behavior remains close to the wide author-reward policy.
```

Latest author market-state alias ablation:

```text
RUN_NAME pattern: piroth2_author_alias_000858_20260425_133749
purpose: test the authors' network.py bug where market_state is replaced by agent_state in fusion
settings: SYMBOL=000858, AUTHOR_MARKET_STATE_ALIAS=true, PPO_INITIAL_SPREAD_BIAS=-0.70, PPO_INITIAL_LOG_STD=-1.8
jobs: 64752007 -> 64752008 -> 64752009
result: C-PPO pnl=-5.83, reward=-2522.75, avg_spread=0.0284, fill_rate=0.135
reference: non-alias 000858 action-fixed medium C-PPO is identical on these metrics; AS baseline pnl=2.25
interpretation: the authors' market-state alias bug is not the current PPO-vs-AS gap driver in this setup.
```

Current pure-PnL diagnostic on Euler:

```text
RUN_NAME pattern: piroth2_purepnl_000858_b0p70_20260425_134135
purpose: isolate the executable author spread penalty by setting REWARD_SPREAD_PENALTY_SCALE=0
settings: SYMBOL=000858, REWARD_MODE=author_pnl, PPO_INITIAL_SPREAD_BIAS=-0.70, PPO_INITIAL_LOG_STD=-1.8
jobs: 64752380 -> 64752381 -> 64752382, pure-PnL baseline=64753703
training result: reward_mean=pnl_mean improved from -13.29 at epoch 1 to +6.93 at epoch 6
C-PPO eval: pnl=+24.72, reward=+24.72, avg_abs_position=11.72, avg_spread=0.0324, fill_rate=0.0684
zero-penalty AS baseline: pnl=+53.67, reward=+53.67, avg_abs_position=358.35, avg_spread=0.0286, fill_rate=0.113
other zero-penalty baselines: Fixed_1 pnl=-90.86, Fixed_2 pnl=-20.70, Fixed_3 pnl=-1.12, Random pnl=-94.26
reference: author-penalty C-PPO pnl=-5.83/reward=-2522.75; AS reference pnl=+2.25 under the author-penalty run
interpretation: zeroing the executable author wide-spread penalty flips PPO from underperforming AS to profitable, but AS still beats PPO on this synthetic 000858 setup under the same zero-penalty objective.
```

Latest reward-penalty scale sweep:

```text
RUN_NAME pattern: piroth2_penaltyscale_000858_s<scale>_20260425_143415
purpose: find whether a less severe version of the executable author spread penalty preserves author intent without destroying PnL
settings: SYMBOL=000858, REWARD_MODE=author_pnl, PPO_INITIAL_SPREAD_BIAS=-0.70, PPO_INITIAL_LOG_STD=-1.8
scale 10 jobs: PPO=64754398, eval=64754399, baseline=64754400
scale 25 jobs: PPO=64754401, eval=64754402, baseline=64754403
scale 50 jobs: PPO=64754404, eval=64754405, baseline=64754406
note: all three reuse the pure-PnL Attn-LOB pretrain checkpoint to avoid redundant pretraining.
baseline results: AS pnl is unchanged at +53.67 across scales; AS reward falls from +22.90 at scale 10 to -23.27 at scale 25 and -100.21 at scale 50.
PPO training through epoch 6: scale 10 pnl_mean=-20.52/reward_mean=-241.82; scale 25 pnl_mean=-28.45/reward_mean=-576.54; scale 50 pnl_mean=-34.12/reward_mean=-1071.46.
PPO eval: scale 10 pnl=-1.98/reward=-236.89/avg_spread=0.0284/fill_rate=0.155
PPO eval: scale 25 pnl=-6.95/reward=-598.37/avg_spread=0.0283/fill_rate=0.161
PPO eval: scale 50 pnl=-6.95/reward=-1189.78/avg_spread=0.0283/fill_rate=0.161
interpretation: lowering the spread penalty helps PPO PnL relative to the author scale=100 path, but AS still dominates on PnL and reward alignment remains poor.
```

Latest larger PPO generalization check:

```text
RUN_NAME pretrain: piroth2_serious_pretrain_000858_20260425_144140
purpose: test whether previous PPO findings are undertraining artifacts by using more data, more epochs, larger batches, and shuffled rollout caps
shared pretrain job: 64754578
settings: SYMBOL=000858, NUM_DAYS=16, TRAIN_DAYS=10, TEST_DAYS=6, EVENTS_PER_DAY_OVERRIDE=60000,
          TORCH_BATCH_SIZE=4096, TORCH_LEARNING_RATE=0.0003, PPO_EPOCHS=24,
          PPO_ROLLOUTS_PER_EPOCH=96, PPO_SHUFFLE_EPISODES=true, PPO_UPDATE_EPOCHS=6,
          PPO_ENTROPY_COEF=0.002, PPO_INITIAL_LOG_STD=-1.6
author-penalty run: piroth2_serious_author_000858_20260425_144140
author-penalty jobs: copy=64754584, PPO=64754585, eval=64754586, baseline=64754587
author-penalty settings: REWARD_SPREAD_PENALTY_SCALE=100, PPO_INITIAL_SPREAD_BIAS=-0.85
author-penalty baselines: AS pnl=+90.77/reward=-221.31, Fixed_1 pnl=-66.10/reward=-108.39, Fixed_2 pnl=-11.97/reward=-1127.08, Random pnl=-58.40/reward=-416.35
author-penalty C-PPO eval: pnl=-164.33/reward=-280.69/avg_abs_position=28.08/avg_spread=0.0122/fill_rate=0.560/turnover=7.50e6
pure-PnL run: piroth2_serious_purepnl_000858_20260425_144140
pure-PnL jobs: copy=64754589, PPO=64754591, eval=64754592, baseline=64754593
pure-PnL settings: REWARD_SPREAD_PENALTY_SCALE=0, PPO_INITIAL_SPREAD_BIAS=-0.70
pure-PnL baselines: AS pnl=reward=+90.77, Fixed_1=-66.10, Fixed_2=-11.97, Fixed_3=-0.08, Random=-58.40
pure-PnL C-PPO eval: pnl=reward=+58.46/avg_abs_position=15.65/avg_spread=0.0339/fill_rate=0.079/turnover=1.04e6
final PPO training: author epoch 24 pnl_mean=-155.07/reward_mean=-393.31; pure-PnL epoch 24 pnl_mean=reward_mean=+46.76
interpretation: more PPO data and compute do help generalization under pure PnL, but the executable author spread penalty drives overtrading and strongly negative PnL. AS remains the best policy on this split.
note: the PPO trainer now shuffles candidate training episodes before applying PPO_ROLLOUTS_PER_EPOCH, so capped training samples across the full training set instead of repeatedly using the earliest episodes.
```

Latest cross-symbol PPO confirmation on Euler:

```text
RUN_NAME pattern: piroth2_xsym_<reward>_<symbol>_20260425_190658
purpose: test whether the 000858 reward/PnL finding generalizes to 000001 and 002415
settings: NUM_DAYS=12, TRAIN_DAYS=8, TEST_DAYS=4, EVENTS_PER_DAY_OVERRIDE=50000,
          TORCH_BATCH_SIZE=4096, TORCH_LEARNING_RATE=0.0003, PPO_EPOCHS=16,
          PPO_ROLLOUTS_PER_EPOCH=80, PPO_SHUFFLE_EPISODES=true, PPO_UPDATE_EPOCHS=6,
          PPO_ENTROPY_COEF=0.002, PPO_INITIAL_LOG_STD=-1.6
000001 shared pretrain: 64770547
000001 author jobs: copy=64770548, PPO=64770549, eval=64770550, baseline=64770551
000001 pure-PnL jobs: copy=64770552, PPO=64770553, eval=64770554, baseline=64770555
002415 shared pretrain: 64770556
002415 author jobs: copy=64770557, PPO=64770558, eval=64770559, baseline=64770560
002415 pure-PnL jobs: copy=64770561, PPO=64770562, eval=64770563, baseline=64770564
status: all jobs completed with exit 0

000001 author C-PPO eval: pnl=-34.32/reward=-86.76/avg_abs_position=18.22/avg_spread=0.0114/fill_rate=0.364/turnover=4.87e5
000001 author baselines: AS pnl=-9.22/reward=-514.89, Fixed_1 pnl=-117.57/reward=-123.07, Fixed_2 pnl=-9.79/reward=-1312.75, Fixed_3 pnl=-0.03/reward=-5553.83, Random pnl=-75.32/reward=-353.03
000001 pure-PnL C-PPO eval: pnl=reward=+5.04/avg_abs_position=3.44/avg_spread=0.0317/fill_rate=0.014/turnover=1.91e4
000001 pure-PnL baselines: AS pnl=reward=-9.22, Fixed_1=-117.57, Fixed_2=-9.79, Fixed_3=-0.03, Random=-75.32

002415 author C-PPO eval: pnl=-47.33/reward=-138.89/avg_abs_position=19.94/avg_spread=0.0113/fill_rate=0.399/turnover=1.36e6
002415 author baselines: AS pnl=-23.76/reward=-577.32, Fixed_1 pnl=-170.14/reward=-176.40, Fixed_2 pnl=-22.83/reward=-942.64, Fixed_3 pnl=-0.75/reward=-5334.96, Random pnl=-101.08/reward=-353.04
002415 pure-PnL C-PPO eval: pnl=reward=+7.54/avg_abs_position=4.59/avg_spread=0.0317/fill_rate=0.023/turnover=7.88e4
002415 pure-PnL baselines: AS pnl=reward=-23.76, Fixed_1=-170.14, Fixed_2=-22.83, Fixed_3=-0.75, Random=-101.08

final PPO training: 000001 author epoch 16 pnl_mean=-54.15/reward_mean=-431.41; 000001 pure-PnL epoch 16 pnl_mean=reward_mean=+3.84; 002415 author epoch 16 pnl_mean=-62.85/reward_mean=-470.33; 002415 pure-PnL epoch 16 pnl_mean=reward_mean=+4.41
interpretation: the author executable spread penalty again teaches C-PPO to quote too tightly, overtrade, and lose PnL while improving shaped reward. Under pure PnL, C-PPO generalizes profitably on both stress symbols and beats AS on PnL, but does so with low fill rates and low inventory. This makes the reward specification, not just PPO undertraining, the main replication blocker.
```

DQN pilot and inventory-guard rerun on Euler:

```text
RUN_NAME author: piroth2_dqnfast_author_000858_20260426_033213
RUN_NAME pure-PnL: piroth2_dqnfast_purepnl_000858_20260426_033213
purpose: start the paper's D-DQN stage with the same reward diagnostic used for C-PPO
shared checkpoint: piroth2_serious_pretrain_000858_20260425_144140/models/attnlob_pretrain.pt
settings: SYMBOL=000858, NUM_DAYS=10, TRAIN_DAYS=6, TEST_DAYS=4, EVENTS_PER_DAY_OVERRIDE=40000,
          TORCH_BATCH_SIZE=1024, TORCH_LEARNING_RATE=0.0003, TORCH_EPOCHS=4,
          MAX_TRAIN_EPISODES_PER_DAY=6, MAX_EVAL_EPISODES_PER_DAY=8,
          DQN_REPLAY_SIZE=100000, DQN_MIN_REPLAY=2048, DQN_UPDATE_INTERVAL=64,
          DQN_TARGET_UPDATE_STEPS=500, DQN_EPSILON_START=0.30, DQN_EPSILON_END=0.03,
          DQN_EPSILON_DECAY=0.82
author jobs: DQN=64793353, eval=64793354, REWARD_SPREAD_PENALTY_SCALE=100
pure-PnL jobs: DQN=64793356, eval=64793357, REWARD_SPREAD_PENALTY_SCALE=0
matching baseline jobs: author=64793897, pure-PnL=64793898
author DQN training: epoch 4 reward_mean=-307.50/pnl_mean=-42.81/loss=0.3146/updates=2228
pure-PnL DQN training: epoch 4 reward_mean=pnl_mean=-44.25/loss=0.1861/updates=2228
author DQN eval: pnl=-82.03/reward=-95.88/avg_abs_position=622.09/avg_spread=0.0215/fill_rate=0.052/turnover=7.61e5
pure-PnL DQN eval: pnl=reward=-90.72/avg_abs_position=828.09/avg_spread=0.0174/fill_rate=0.087/turnover=1.27e6
author baselines on same split: AS pnl=+58.94/reward=-233.69, Fixed_1 pnl=-81.16/reward=-113.38,
                                Fixed_2 pnl=-16.78/reward=-1106.47, Fixed_3 pnl=-0.78/reward=-5630.28,
                                Random pnl=-71.69/reward=-433.25
pure-PnL baselines on same split: AS pnl=reward=+58.94, Fixed_1=-81.16, Fixed_2=-16.78,
                                  Fixed_3=-0.78, Random=-71.69
interpretation: the bounded DQN pilot is functional but not competitive. It learns a high-inventory,
                high-turnover policy and does not reproduce the PPO pure-PnL recovery. On this
                split DQN loses to AS by roughly 141-150 PnL points and also underperforms the
                random baseline on PnL. A paper-faithful final DQN run needs DQN-specific tuning
                before it is worth scaling.
note: this is intentionally a bounded DQN pilot, not the final DQN replication run.
note: earlier DQN pilots were cancelled after 5.5 hours because the update cadence was too slow
      (batch 4096, update every 4 env steps, 20 episodes/day). DQN now writes per-epoch history
      and log lines so future runs are observable before completion.
paper-alignment fix: the authors' `env_discrete.py:action2order` prevents adding to inventory beyond
                     +/-10 TRADE_UNIT by suppressing the side that would increase the breach. The
                     first bounded DQN pilot missed this guard, which explains its huge inventory.
                     `DiscreteActionPolicy` now applies the same guard, with a focused regression
                     test in `tests/test_piroth2_paper_env.py`.
fixed rerun: author RUN_NAME=piroth2_dqnfix_author_000858_20260426_093612, jobs DQN=64801641 -> eval=64801642
             pure-PnL RUN_NAME=piroth2_dqnfix_purepnl_000858_20260426_093612, jobs DQN=64801643 -> eval=64801645
             focused Euler test before submission: tests/test_piroth2_paper_env.py, 7 passed
fixed author DQN training: epoch 4 reward_mean=-294.97/pnl_mean=-53.94/loss=0.4021/updates=2228
fixed pure-PnL DQN training: epoch 4 reward_mean=pnl_mean=-45.72/loss=0.1840/updates=2228
fixed author DQN eval: pnl=-90.00/reward=-132.19/avg_abs_position=828.55/avg_spread=0.0168/fill_rate=0.089/turnover=1.29e6
fixed pure-PnL DQN eval: pnl=reward=-88.16/avg_abs_position=830.97/avg_spread=0.0158/fill_rate=0.093/turnover=1.35e6
fixed-rerun interpretation: the author inventory guard is now present, but it does not by itself
                            make D-DQN paper-competitive. It prevents adding to a breached side;
                            it does not force liquidation. The learned policy still carries very
                            large inventory and remains far behind AS (+58.94 PnL) and prior pure-PnL
                            C-PPO (+58.46 PnL on the larger 000858 check). The next useful DQN work
                            is action/inventory diagnostics and DQN-specific tuning, not scaling this
                            exact configuration.
DQN action/inventory diagnostic:
author diagnostic jobs: first 64803406 failed due sbatch --wrap shell startup; replacement 64803904 completed
pure-PnL diagnostic jobs: first 64803407 failed due sbatch --wrap shell startup; replacement 64803905 completed
author action fractions: a0=58.6%, a6=11.6%, a5=11.0%, a2=8.7%, a4=7.6%, a7(liquidate)=0.19%
pure-PnL action fractions: a0=72.4%, a1=8.3%, a2=7.6%, a5=5.0%, a4=4.8%, a7(liquidate)=0.10%
author breach_before_frac=62.6%, liquidation_rate_when_breached=0.0%, mean_q_margin_liquidation_vs_selected=-6.79
pure-PnL breach_before_frac=63.5%, liquidation_rate_when_breached=0.0%, mean_q_margin_liquidation_vs_selected=-6.53
diagnostic interpretation: D-DQN is not merely suffering from missing action guards. Once it reaches
                           the inventory boundary, it strongly prefers ordinary quote actions over
                           action 7 liquidation. That points to reward/target credit assignment,
                           exploration, or action design as the DQN-specific blocker.
forced-liquidation counterfactual:
author force-liquidate job=64804953, pure-PnL force-liquidate job=64804954
author force-liquidate eval: pnl=-30.50/reward=-157.50/avg_abs_position=421.12/avg_spread=0.0221/fill_rate=0.216/turnover=4.56e6
pure-PnL force-liquidate eval: pnl=reward=-32.78/avg_abs_position=426.83/avg_spread=0.0214/fill_rate=0.226/turnover=4.83e6
delta vs standard: author pnl +59.50 and avg_abs_position -407.43; pure-PnL pnl +55.38 and avg_abs_position -404.13
counterfactual interpretation: failure to exit breached inventory explains a large part of DQN loss,
                               but not all of it. Forced liquidation beats Random and Fixed_1 on the
                               reduced split, but still remains far behind AS (+58.94). The remaining
                               gap is bad entry/quote selection and credit assignment, so the next
                               bounded DQN diagnostic should add explicit inventory-risk reward.
inventory-penalty DQN diagnostic:
RUN_NAME=piroth2_dqn_invpen_000858_20260426_112101
jobs: DQN=64805433 -> eval=64805439, action diagnostic=64805994
settings: same reduced 000858 split, REWARD_MODE=hybrid, REWARD_SPREAD_PENALTY_SCALE=0,
          REWARD_ZETA=0.05, REWARD_USE_DAMPENED_PNL=false, REWARD_USE_TRADING_PNL=false,
          REWARD_USE_INVENTORY_PENALTY=true, DQN_EPSILON_START=0.35, DQN_EPSILON_END=0.05
training: epoch 4 reward_mean=-70.45/pnl_mean=+22.81/loss=0.1047/updates=2228
eval: pnl=+36.94/reward=-60.25/avg_abs_position=94.39/avg_spread=0.0273/fill_rate=0.135/turnover=2.19e6
interpretation: explicit inventory-risk reward turns DQN from a broken high-inventory policy into a
                profitable low-inventory policy on the reduced 000858 split. It still trails AS
                (+58.94) and is a diagnostic deviation from the executable author reward, not the
                paper-faithful final DQN result.
inventory-penalty action diagnostic: job=64805994
action fractions: a0=35.3%, a2=17.3%, a4=15.6%, a1=9.8%, a7(liquidate)=9.4%, a5=5.7%, a3=3.8%, a6=3.2%
breach_before_frac=0.0%, liquidation_rate_all=9.4%, mean_q_margin_liquidation_vs_selected=-0.41
diagnostic interpretation: unlike the paper-reward DQN, the inventory-penalty DQN avoids the guard
                           boundary entirely and gives liquidation a competitive Q-value. This confirms
                           that the DQN implementation can learn inventory-aware behavior when the
                           reward supplies timely inventory-risk feedback.
larger inventory-penalty confirmation:
RUN_NAME=piroth2_dqn_invpen_confirm_000858_20260426_115758
jobs: DQN=64806549 -> eval=64806550, action diagnostic=64809365
settings: NUM_DAYS=16, TRAIN_DAYS=10, TEST_DAYS=6, EVENTS_PER_DAY_OVERRIDE=60000,
          TORCH_EPOCHS=8, MAX_TRAIN_EPISODES_PER_DAY=8, MAX_EVAL_EPISODES_PER_DAY=10,
          same inventory-penalty reward as the reduced diagnostic
training: epoch 8 reward_mean=+22.40/pnl_mean=+41.07/loss=0.0491/updates=10141
eval: pnl=+52.13/reward=+33.16/avg_abs_position=33.59/avg_spread=0.0311/fill_rate=0.094/turnover=1.31e6
eval robustness: 60/60 held-out episodes positive, pnl_min=+11.00, pnl_max=+113.00
confirmation interpretation: the inventory-penalty diagnostic scales to the larger 000858 split and
                             nearly matches serious pure-PnL C-PPO (+58.46), while keeping much lower
                             inventory than AS. It still trails AS on this symbol (+90.77) and remains
                             a diagnostic reward, not the paper-faithful D-DQN result.
cross-symbol inventory-penalty confirmation:
RUN_NAME 000001=piroth2_dqn_invpen_xsym_000001_20260426_133449, jobs DQN=64810212 -> eval=64810216
RUN_NAME 002415=piroth2_dqn_invpen_xsym_002415_20260426_133449, jobs DQN=64810218 -> eval=64810220
settings: NUM_DAYS=12, TRAIN_DAYS=8, TEST_DAYS=4, EVENTS_PER_DAY_OVERRIDE=50000,
          TORCH_EPOCHS=6, MAX_TRAIN_EPISODES_PER_DAY=8, MAX_EVAL_EPISODES_PER_DAY=10,
          same inventory-penalty reward as the 000858 diagnostics
000001 training: epoch 6 reward_mean=-0.62/pnl_mean=+13.97/loss=0.0215
000001 eval: pnl=+19.23/reward=+3.21/avg_abs_position=24.53/avg_spread=0.0279/fill_rate=0.083/turnover=1.16e5
000001 eval robustness: 40/40 held-out episodes positive, pnl_min=+4.00, pnl_max=+52.00
000001 comparison: pure-PnL C-PPO=+5.04, AS=-9.22, author C-PPO=-34.32
002415 training: epoch 6 reward_mean=+6.37/pnl_mean=+16.44/loss=0.0199
002415 eval: pnl=+24.08/reward=+13.38/avg_abs_position=18.10/avg_spread=0.0261/fill_rate=0.065/turnover=2.27e5
002415 eval robustness: 40/40 held-out episodes positive, pnl_min=+9.00, pnl_max=+53.00
002415 comparison: pure-PnL C-PPO=+7.54, AS=-23.76, author C-PPO=-47.33
cross-symbol interpretation: inventory-penalty DQN generalizes on the two stress symbols and beats
                             both AS and pure-PnL C-PPO there. This is useful implementation evidence,
                             but it remains a diagnostic reward change rather than the paper's
                             executable D-DQN objective.
paper-faithful tuned DQN check:
RUN_NAME=piroth2_dqn_author_tuned_000858_20260426_142330
jobs: DQN=64812670 -> eval=64812671, action diagnostic=64821289
settings: REWARD_MODE=author_pnl, REWARD_SPREAD_PENALTY_SCALE=100, NUM_DAYS=16,
          TRAIN_DAYS=10, TEST_DAYS=6, EVENTS_PER_DAY_OVERRIDE=60000, TORCH_EPOCHS=10,
          TORCH_LEARNING_RATE=0.00025, DQN_REPLAY_SIZE=250000, DQN_UPDATE_INTERVAL=96,
          DQN_TARGET_UPDATE_STEPS=1000, DQN_EPSILON_START=0.50, DQN_EPSILON_END=0.05
training: epoch 10 reward_mean=-290.25/pnl_mean=-87.99/loss=0.1867/updates=8435
eval: pnl=-120.43/reward=-131.12/avg_abs_position=627.81/avg_spread=0.0180/fill_rate=0.061/turnover=9.23e5
eval robustness: 4/60 held-out episodes positive, pnl_min=-283.00, pnl_max=+44.00
action diagnostic: action fractions a7=21.9%, a4=20.1%, a5=16.6%, a0=15.6%, a2=8.5%, a1=8.4%, a3=6.1%, a6=2.7%
                   breach_before_frac=40.1%, liquidation_rate_when_breached=0.0%,
                   mean_q_margin_liquidation_vs_selected=-7.21 when breached and -4.76 overall
tuned interpretation: more exploration and slower DQN updates did not rescue the paper reward.
                      The policy still carries very high inventory and underperforms standard
                      author DQN, serious author C-PPO, AS, and the diagnostic inventory-penalty
                      DQN. The action diagnostic confirms the same non-liquidating breached-inventory
                      failure mode seen in the fixed DQN runs, despite frequent liquidation actions
                      while not breached.
```

Latest result summary artifact:

```text
/cluster/project/math/piroth/mlfcs-gapa/artifacts_piroth2/paper_results_summary_20260424_231500/summary.md
/cluster/project/math/piroth/mlfcs-gapa/artifacts_piroth2/paper_results_summary_latest/summary.md
/cluster/project/math/piroth/mlfcs-gapa/artifacts_piroth2/paper_results_summary_latest/index.html
tables=medium_agents, medium_baselines, latency, ablations, ppo_seed_sweep, paired_seed_baselines, ppo_seed_vs_as, ppo_tuned, ppo_tuned_vs_as, synthetic_validation
```

Real-data support and active runs:

```text
data_source=real is now supported through piroth/real_data.py.
Euler real-data root: /cluster/work/math/piroth/mlfcs-gapa/data/processed
expected layout: {AAPL,GOOGL}/YYYYMMDD/{ask,bid,price,trades,msg}.csv
bounded real-run settings: REAL_EVENT_STRIDE=100, EVENTS_PER_DAY_OVERRIDE=60000,
                           TRAIN_DAYS=6, TEST_DAYS=3, EPISODE_LENGTH=2000

real PPO/eval status:
AAPL author:   piroth2_real_author_AAPL_20260427_105334, jobs 64880911 -> 64880913 -> 64880916, baseline 64880918
AAPL pure-PnL: piroth2_real_purepnl_AAPL_20260427_105334, jobs 64880919 -> 64880922 -> 64880924, baseline 64880926
GOOGL author:  piroth2_real_author_GOOGL_20260427_105334, jobs 64880929 -> 64880931 -> 64880933, baseline 64880936
GOOGL pure-PnL: piroth2_real_purepnl_GOOGL_20260427_105334, jobs 64880938 -> 64880940 -> 64880942, baseline 64880945

completed real baselines:
AAPL author quality score=63.85, flags=high trade density/overly directional windows/low order-flow persistence;
  AS pnl=0.00 reward=-382.38 fill_rate=0.000, Fixed_1 pnl=-30.67, Fixed_2 pnl=-1.63, Fixed_3 pnl=-2.33, Random pnl=-27.29
AAPL pure-PnL quality score=63.85, same quality flags;
  AS pnl=0.00 reward=0.00 fill_rate=0.000, Fixed_1 pnl=-30.67, Fixed_2 pnl=-1.63, Fixed_3 pnl=-2.33, Random pnl=-27.29
GOOGL author quality score=64.27, same quality flags;
  AS pnl=-0.33 reward=-407.67 fill_rate=0.004, Fixed_1 pnl=-9.04, Fixed_2 pnl=-4.29, Fixed_3 pnl=-13.04, Random pnl=-22.29
GOOGL pure-PnL quality score=64.27, same quality flags;
  AS pnl=-0.33 reward=-0.33 fill_rate=0.004, Fixed_1 pnl=-9.04, Fixed_2 pnl=-4.29, Fixed_3 pnl=-13.04, Random pnl=-22.29
AAPL author C-PPO eval: pnl=-18.29/reward=-44.33/avg_abs_position=40.09/avg_spread=0.0229/fill_rate=0.696/turnover=5.23e5, positive episodes=9/24
AAPL pure-PnL C-PPO eval: pnl=reward=-17.08/avg_abs_position=39.20/avg_spread=0.0248/fill_rate=0.691/turnover=5.19e5, positive episodes=9/24
GOOGL author C-PPO eval: pnl=-14.21/reward=-41.38/avg_abs_position=44.75/avg_spread=0.0253/fill_rate=0.742/turnover=5.25e5, positive episodes=11/24
GOOGL pure-PnL C-PPO eval: pnl=reward=-7.38/avg_abs_position=44.74/avg_spread=0.0350/fill_rate=0.596/turnover=4.17e5, positive episodes=10/24
interpretation: real-data path works, but REAL_EVENT_STRIDE=100 still produces high-trade-density,
                highly directional 2000-event windows. Final real-data conclusions need better event
                sampling or episode construction before comparing directly to the synthetic paper-scale runs.
                On AAPL and GOOGL, C-PPO trades actively and usually loses less than the worst active
                baselines, but it still trails AS and the wider fixed baselines on PnL. Pure PnL helps
                GOOGL relative to author reward, but does not make the current real split competitive.
```

Synthetic generator variant sweep:

```text
new opt-in simulator knobs:
  ORDER_FLOW_MEMORY: repeats previous noise-taker direction to increase sign persistence
  VOLATILITY_CLUSTER_STRENGTH/PERSISTENCE: temporarily raises fair-value noise after shocks

completed author-reward generator variants on 000858:
flowvol:  piroth2_synth_flowvol_author_000858_20260427_105410, jobs 64881054 -> 64881056 -> 64881058, baseline 64881060
          ORDER_FLOW_MEMORY=0.35, VOLATILITY_CLUSTER_STRENGTH=0.45, VOLATILITY_CLUSTER_PERSISTENCE=0.992
          quality score=99.63, flags=0, AS pnl=+94.22/reward=-153.34
          C-PPO eval pnl=-191.50/reward=-339.70/avg_abs_position=30.02/avg_spread=0.0128/fill_rate=0.597/turnover=7.89e6
          interpretation: richer order-flow and volatility clustering produce high-quality synthetic statistics,
                          but the author spread-penalty PPO still learns tight, high-fill overtrading and loses
                          to AS by roughly 286 PnL points.
noinform: piroth2_synth_noinform_author_000858_20260427_105410, jobs 64881063 -> 64881065 -> 64881067, baseline 64881069
          INFORMED_MARKET_ORDER_PROB=0.0, NOISE_MARKET_ORDER_PROB=0.48
          quality score=71.77, flag=median 2000-event window is too flat, AS pnl=+2.72/reward=-2112.00
          C-PPO eval pnl=-0.97/reward=-166.90/avg_abs_position=13.53/avg_spread=0.0114/fill_rate=0.271/turnover=3.57e6
          interpretation: removing informed flow makes the market too flat for a useful replication test; it lowers
                          PPO PnL damage but mainly because the price process is near static.
```

Latest neural smoke on Euler:

```text
RUN_NAME=piroth2_neural_smoke_tight_20260424_194000
EPISODE_LENGTH=500
pretrain_accuracy=0.7339
C-PPO produced fills in all eval episodes; mean pnl=-14.5 across 4 episodes
D-DQN produced fills in all eval episodes; mean pnl=10.75 across 4 episodes
```

Latest medium neural comparison on Euler:

```text
RUN_NAME pattern: piroth2_gpu_medium_<symbol>_20260424_214500 / 223500
settings: GPU training, TORCH_BATCH_SIZE=1024, EVENTS_PER_DAY_OVERRIDE=30000, EPISODE_LENGTH=2000
000001: C-PPO pnl=-4.92, D-DQN pnl=-146.08, AS baseline pnl=-7.08
000858: C-PPO pnl=74.67, D-DQN pnl=-105.33, AS baseline pnl=71.08
002415: C-PPO pnl=3.50, D-DQN pnl=-139.33, AS baseline pnl=4.83
note: PPO initial action log-std is -1.5; the earlier std=1.0 setting made medium PPO rollouts much too noisy.
```

Current PPO robustness sweep on Euler:

```text
RUN_NAME pattern: piroth2_ppo_seed<seed>_<symbol>_20260424_232000
symbols=000001,000858,002415
seeds=11,17
stages=pretrain -> train-ppo -> evaluate-ppo completed
000001 seed 11/17: pnl=0.50, -27.08
000858 seed 11/17: pnl=52.08, 57.50
002415 seed 11/17: pnl=-54.25, -106.83
interpretation: PPO robustness is acceptable on 000858, borderline on 000001, and currently weak on 002415.
```

Matched seed AS baselines:

```text
RUN_NAME pattern: piroth2_baseline_seed<seed>_<symbol>_20260425_000500
000001 seed 11/17 AS pnl=3.75, -21.00
000858 seed 11/17 AS pnl=81.25, 82.42
002415 seed 11/17 AS pnl=-57.25, -63.83
paired interpretation: current PPO does not consistently beat AS on matched synthetic seeds; this is now tracked in ppo_seed_vs_as.
```

Current focused 002415 PPO tune on Euler:

```text
RUN_NAME pattern: piroth2_ppo_tuned_002415_seed<seed>_20260424_235000
seeds=11,17
changes vs medium seed sweep: MODE=full with 8/4/4 days preserved, PPO_EPOCHS=8, PPO_ROLLOUTS_PER_EPOCH=16, PPO_UPDATE_EPOCHS=4, PPO_ENTROPY_COEF=0.002
jobs seed 11: 64712984 -> 64712985 -> 64712988
jobs seed 17: 64712989 -> 64712990 -> 64712991
dependent summary refresh: 64714345
```

Current PPO optimization smoke on Euler:

```text
RUN_NAME=piroth2_ppoopt_smoke_000001_20260425_001500
purpose=validate single-stack PPO minibatch update path after training-speed cleanup
jobs=64713767 -> 64713768 -> 64713769
result=4 eval episodes, mean pnl=-12.50, mean reward=-199.43
```

Latest fast cross-symbol baseline sweep on Euler:

```text
RUN_NAME pattern: piroth2_baselines_fast_<symbol>_20260424_201000
000001: quality=99.41, AS pnl=-34.0, Fixed_1 pnl=-142.0, Fixed_2 pnl=-21.0, Random pnl=-133.0
000858: quality=99.83, AS pnl=119.5, Fixed_1 pnl=5.0, Fixed_2 pnl=-9.0, Random pnl=-26.0
002415: quality=99.26, AS pnl=-10.5, Fixed_1 pnl=-41.0, Fixed_2 pnl=-14.0, Random pnl=-48.0
```

The exact 30k-event medium baseline comparison should be profiled before rerun.
The original path spent most time inside pandas-heavy AS calibration; after the
array calibration rewrite, the exact medium baseline comparison completes in
about 1.5 minutes per symbol.

Latest latency sweep:

```text
RUN_NAME pattern: piroth2_latency_fast_<symbol>_20260424_223500
AS pnl by latency 0/1/5/10/20:
000001: -8.0, -34.0, -84.5, -117.0, -135.0
000858: 129.0, 119.5, 97.5, 81.0, 65.0
002415: 1.0, -10.5, -38.5, -52.5, -51.0
```

Latest 000858 PPO/DQN ablation sweep:

```text
RUN_NAME pattern: piroth2_ppo_ablation_<variant>_000858_20260424_224500
variant        C-PPO pnl    D-DQN pnl
full              56.08       -62.42
no_lob            61.00       -46.58
no_dynamic        72.50        -0.08
no_pretrain       36.83      -102.17
```

Latest speed profile:

```text
profile-final-gen1: 30k-event day generation 14.45s under cProfile
profile-final-gen4: four 30k-event days 53.45s under cProfile
profile-calib-base1: baseline profile 67.03s after AS calibration rewrite, down from 332.99s
```
