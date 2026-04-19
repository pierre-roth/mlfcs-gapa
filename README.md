# piroth

Fresh branch for a paper-driven, simulated-data reimplementation of the continuous market-making agent from [paper.tex](/Users/piroth/Documents/projects/mlfcs-gapa/paper.tex).

What is included:
- `piroth/`: synthetic simulator, paper-style data/features, Attn-LOB pretraining, continuous PPO environment/training, and reporting
- `cluster/`: minimal Euler runner and one sbatch entrypoint for the full suite
- `tests/`: simulator, reward/env, and end-to-end smoke coverage

Main assumptions:
- continuous agent only
- synthetic data purpose-built for the paper setup
- action space follows the paper: `A1, A2 in [0, 1]`, inventory-directed reservation bias, single bid/ask quote, `2000`-event episodes, terminal liquidation
- reward follows the paper: `DP + TP - IP`

Local commands:

```bash
uv run python -m piroth.simulator --mode smoke
uv run python -m piroth.run_suite --mode smoke --run-name piroth_smoke
uv run pytest tests/test_piroth_simulator.py tests/test_piroth_env.py tests/test_piroth_smoke.py -q
```

Euler:

```bash
bash cluster/submit_piroth_suite.sh
```
