from __future__ import annotations

import pandas as pd
import pyrallis

from .config import RLTrainConfig
from .pipeline import evaluate_baseline_policy, load_symbol_splits, prepare_run, resolve_symbol_rl_config, save_episode_results, standard_baselines, summarize_results
from .utils import ensure_dir, save_json


def run_evaluation(config: RLTrainConfig) -> dict[str, dict[str, dict[str, float]]]:
    config.apply_mode_defaults()
    out_dir = prepare_run(config, label="evaluate")
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for symbol in config.symbols:
        splits = load_symbol_splits(config, symbol)
        symbol_cfg = resolve_symbol_rl_config(config, splits["train"])
        symbol_dir = ensure_dir(out_dir / symbol / "evaluation")
        symbol_summary: dict[str, dict[str, float]] = {}
        for baseline in standard_baselines(symbol_cfg):
            results, runtime = evaluate_baseline_policy(baseline, splits["test"], symbol_cfg)
            frame = save_episode_results(symbol_dir / f"{baseline.name}.csv", results)
            baseline_summary = summarize_results(frame)
            baseline_summary["episode_length"] = int(symbol_cfg.episode_length)
            baseline_summary.update(runtime)
            save_json(symbol_dir / f"{baseline.name}_timing.json", runtime)
            symbol_summary[baseline.name] = baseline_summary
        save_json(symbol_dir / "summary.json", symbol_summary)
        summaries[symbol] = symbol_summary
    save_json(out_dir / "evaluation_summary.json", summaries)
    return summaries


@pyrallis.wrap()
def main(config: RLTrainConfig) -> None:
    run_evaluation(config)


if __name__ == "__main__":
    main()
