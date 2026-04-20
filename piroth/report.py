from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyrallis
import torch

from .baselines import AvellanedaStoikovPolicy, FixedLevelPolicy, calibrate_avellaneda_stoikov
from .config import ReportConfig
from .data import load_splits
from .env import ContinuousMarketEnv
from .train import _build_model, evaluate_model, summarize
from .utils import ensure_dir, save_json


def _evaluate_baseline(policy, days, config: ReportConfig) -> dict[str, float]:
    results = []
    for day in days:
        env = ContinuousMarketEnv(day, config)
        for episode_index, span in enumerate(env.selected_episodes(config.max_eval_episodes_per_day)):
            obs = env.reset(span)
            done = False
            while not done:
                quote_idx = max(int(env.episode_decisions[env.cursor] - env.config.latency), env.config.lookback - 1)
                decision = policy.act(day, quote_idx, env.inventory, env.cursor, len(env.episode_decisions))
                obs, _, done, _ = env.step(
                    {
                        "ask_price": decision.ask_price,
                        "ask_volume": decision.ask_volume,
                        "bid_price": decision.bid_price,
                        "bid_volume": decision.bid_volume,
                        "spread": decision.spread,
                        "reservation": 0.5 * (decision.ask_price + decision.bid_price),
                    }
                )
            results.append(env.episode_result(policy.name, episode_index))
    return summarize(pd.DataFrame(results))


def run_report(config: ReportConfig) -> dict[str, dict[str, float]]:
    config.apply_mode_defaults()
    summaries = {}
    for symbol in config.symbols:
        splits = load_splits(config, symbol)
        model = _build_model(config, symbol)
        model.load_state_dict(torch.load(Path(config.output_dir()) / symbol / "ppo" / "model.pt", map_location="cpu"))
        symbol_dir = ensure_dir(Path(config.output_dir()) / symbol / "report")
        ppo_results = evaluate_model([ContinuousMarketEnv(day, config) for day in splits["test"]], model, config)
        ppo_frame = pd.DataFrame(ppo_results)
        ppo_frame.to_csv(symbol_dir / "ppo_episodes.csv", index=False)
        summary = summarize(ppo_frame)
        baselines = {
            "Fixed_1": _evaluate_baseline(FixedLevelPolicy(config, 1), splits["test"], config),
            "Fixed_2": _evaluate_baseline(FixedLevelPolicy(config, 2), splits["test"], config),
            "AS": _evaluate_baseline(AvellanedaStoikovPolicy(config, calibrate_avellaneda_stoikov(splits["train"], config)), splits["test"], config),
        }
        summary["fixed1_pnl_mean"] = float(baselines["Fixed_1"].get("pnl_mean", 0.0))
        summary["as_pnl_mean"] = float(baselines["AS"].get("pnl_mean", 0.0))
        save_json(symbol_dir / "summary.json", summary)
        save_json(symbol_dir / "baselines.json", baselines)
        summaries[symbol] = summary
    save_json(Path(config.output_dir()) / "report_summary.json", summaries)
    return summaries


@pyrallis.wrap()
def main(config: ReportConfig) -> None:
    run_report(config)


if __name__ == "__main__":
    main()
