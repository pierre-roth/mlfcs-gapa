# piroth2

This branch is focused entirely on building a good synthetic LOB simulator before revisiting RL.

Current scope:
- on-the-fly event generation
- explicit FIFO order book
- agent-based market dynamics
- export in the paper-compatible `ask/bid/price/trades/msg` format
- Avellaneda-Stoikov baseline replay
- diagnostics and plots for midprice and LOB behavior

Main files:
- [paper.tex](/Users/piroth/Documents/projects/mlfcs-gapa/paper.tex)
- [docs/piroth2_simulator.md](/Users/piroth/Documents/projects/mlfcs-gapa/docs/piroth2_simulator.md)
- [piroth/config.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/config.py)
- [piroth/simulator.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/simulator.py)
- [piroth/baselines.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/baselines.py)
- [piroth/diagnostics.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/diagnostics.py)

Cluster diagnostics:

```bash
bash cluster/submit_piroth2_diagnostics.sh
```

This branch does not pre-generate the full dataset. Days are generated deterministically on demand and can optionally be exported for inspection.
