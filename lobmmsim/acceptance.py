from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyrallis

from .config import ExperimentConfig, GenerateConfig, RLTrainConfig
from .pipeline import evaluate_baseline_policy, load_symbol_splits, standard_baselines
from .simulator import generate_dataset
from .utils import ensure_dir, save_json


def run_acceptance_check(config: ExperimentConfig, symbol: str | None = None, seeds: list[int] | None = None) -> dict[str, object]:
    symbol = symbol or config.symbols[0]
    seeds = list(seeds or config.acceptance_seeds)
    root = ensure_dir(Path(config.output_dir()) / "acceptance")
    rows = []
    for seed in seeds:
        data_dir = root / f"seed_{seed}" / "data"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        gen_cfg = GenerateConfig(**{**config.__dict__, "symbols": [symbol], "data_dir": str(data_dir), "seed": seed})
        gen_cfg.apply_mode_defaults()
        generate_dataset(gen_cfg)
        rl_cfg = RLTrainConfig(**{**config.__dict__, "symbols": [symbol], "data_dir": str(data_dir), "seed": seed})
        rl_cfg.apply_mode_defaults()
        splits = load_symbol_splits(rl_cfg, symbol)
        for baseline in standard_baselines(rl_cfg):
            results, _ = evaluate_baseline_policy(baseline, splits["test"], rl_cfg)
            rows.append(
                {
                    "seed": seed,
                    "baseline": baseline.name,
                    "pnl_mean": float(np.mean([result.pnl for result in results])) if results else 0.0,
                    "reward_mean": float(np.mean([result.reward for result in results])) if results else 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "summary.csv", index=False)
    fixed = frame[frame["baseline"] == "Fixed_1"]["pnl_mean"].to_numpy(dtype=np.float64)
    oracle = frame[frame["baseline"] == "OraclePaper"]["pnl_mean"].to_numpy(dtype=np.float64)
    summary = {
        "symbol": symbol,
        "seeds": seeds,
        "fixed1_pnl_mean_avg": float(fixed.mean()) if fixed.size else 0.0,
        "oracle_paper_pnl_mean_avg": float(oracle.mean()) if oracle.size else 0.0,
        "fixed1_positive_seed_fraction": float((fixed > 0).mean()) if fixed.size else 0.0,
        "oracle_better_than_fixed_fraction": float((oracle >= fixed).mean()) if fixed.size and oracle.size == fixed.size else 0.0,
    }
    save_json(root / "summary.json", summary)
    return summary


@pyrallis.wrap()
def main(config: ExperimentConfig) -> None:
    run_acceptance_check(config)


if __name__ == "__main__":
    main()
