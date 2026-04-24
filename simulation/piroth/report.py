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


def _interpretability_summary(traces: list[pd.DataFrame]) -> dict[str, float]:
    if not traces:
        return {}
    frame = pd.concat(traces, ignore_index=True)
    metrics: dict[str, float] = {}
    if "quote_bias" in frame and "latent_alpha" in frame and frame["quote_bias"].std(ddof=0) > 0 and frame["latent_alpha"].std(ddof=0) > 0:
        metrics["bias_alpha_corr"] = float(frame["quote_bias"].corr(frame["latent_alpha"]))
    if "event_actor" in frame:
        metrics["informed_event_share"] = float((frame["event_actor"] == "informed_taker").mean())
        metrics["noise_event_share"] = float((frame["event_actor"] == "noise_taker").mean())
        metrics["maker_event_share"] = float((frame["event_actor"] == "competing_mm").mean())
    if "maker_agent" in frame:
        metrics["competing_mm_context_share"] = float((frame["maker_agent"] == "competing_mm").mean())
        metrics["liquidity_provider_context_share"] = float((frame["maker_agent"] == "liquidity_provider").mean())
    if "queue_pressure" in frame:
        metrics["queue_pressure_mean"] = float(frame["queue_pressure"].mean())
    if "top_imbalance" in frame:
        metrics["top_imbalance_mean"] = float(frame["top_imbalance"].mean())
    # Spread dynamics diagnostics
    if "spread_ticks" in frame:
        metrics["spread_mean"] = float(frame["spread_ticks"].mean())
        metrics["spread_std"] = float(frame["spread_ticks"].std(ddof=0))
        metrics["spread_gt1_frac"] = float((frame["spread_ticks"] > 1.5).mean())
    # Volatility clustering diagnostics
    if "vol_state" in frame:
        metrics["vol_state_mean"] = float(frame["vol_state"].mean())
        metrics["vol_state_std"] = float(frame["vol_state"].std(ddof=0))
    return metrics


def run_report(config: ReportConfig) -> dict[str, dict[str, float]]:
    config.apply_mode_defaults()
    summaries = {}
    for symbol in config.symbols:
        splits = load_splits(config, symbol)
        model = _build_model(config, symbol)
        model.load_state_dict(torch.load(Path(config.output_dir()) / symbol / "ppo" / "model.pt", map_location="cpu"))
        symbol_dir = ensure_dir(Path(config.output_dir()) / symbol / "report")
        ppo_results, traces = evaluate_model([ContinuousMarketEnv(day, config) for day in splits["test"]], model, config, collect_traces=True)
        ppo_frame = pd.DataFrame(ppo_results)
        ppo_frame.to_csv(symbol_dir / "ppo_episodes.csv", index=False)
        summary = summarize(ppo_frame)
        summary.update(_interpretability_summary(traces))
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