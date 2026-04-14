from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd

from .baselines import BaselinePolicy, FixedLevelPolicy, OracleAlphaPolicy
from .config import ExperimentConfig, RLTrainConfig
from .data import DayData, apply_lob_normalizer, discover_days, fit_lob_normalizer, load_day_data, split_days
from .env import MarketMakingEnv
from .metrics import EpisodeResult, sharpe
from .utils import ensure_dir, save_json, set_seed, timestamped_name


def prepare_run(config: ExperimentConfig, label: str | None = None) -> Path:
    set_seed(config.seed)
    if not config.run_name:
        config.run_name = timestamped_name(config.mode)
    out = config.output_dir()
    ensure_dir(out)
    save_json(out / "config.json", config)
    if label:
        save_json(out / f"config_{label}.json", config)
    return out


def load_symbol_splits(config: ExperimentConfig, symbol: str) -> dict[str, list[DayData]]:
    days = discover_days(config.data_dir, symbol)
    train_days, val_days, test_days = split_days(days, config.train_days, config.val_days, config.test_days)
    out = {
        "train": [load_day_data(symbol, day, config) for day in train_days],
        "val": [load_day_data(symbol, day, config) for day in val_days],
        "test": [load_day_data(symbol, day, config) for day in test_days],
    }
    normalizer = fit_lob_normalizer(out["train"])
    for split_days_data in out.values():
        for day in split_days_data:
            apply_lob_normalizer(day, normalizer)
    return out


def save_episode_results(path: str | Path, results: Iterable[EpisodeResult]) -> pd.DataFrame:
    frame = pd.DataFrame([result.to_dict() for result in results])
    ensure_dir(Path(path).parent)
    frame.to_csv(path, index=False)
    return frame


def summarize_results(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "episodes": float(len(frame)),
        "pnl_mean": float(frame["pnl"].mean()) if not frame.empty else 0.0,
        "nd_pnl_mean": float(frame["nd_pnl"].mean()) if not frame.empty else 0.0,
        "pnl_map_mean": float(frame["pnl_map"].mean()) if not frame.empty else 0.0,
        "profit_ratio_mean": float(frame["profit_ratio"].mean()) if not frame.empty else 0.0,
        "reward_mean": float(frame["reward"].mean()) if not frame.empty else 0.0,
        "turnover_mean": float(frame["turnover"].mean()) if not frame.empty else 0.0,
        "trades_mean": float(frame["trades"].mean()) if not frame.empty else 0.0,
        "fill_rate_mean": float(frame["fill_rate"].mean()) if not frame.empty else 0.0,
        "avg_bias_mean": float(frame["avg_bias"].mean()) if not frame.empty else 0.0,
        "alpha_bias_corr_mean": float(frame["alpha_bias_corr"].mean()) if not frame.empty else 0.0,
        "sharpe": sharpe(frame["pnl"].tolist()) if not frame.empty else 0.0,
    }


def standard_baselines(config: ExperimentConfig) -> list[BaselinePolicy]:
    return [FixedLevelPolicy(config, 1), OracleAlphaPolicy(config)]


def evaluate_baseline_policy(policy: BaselinePolicy, days: list[DayData], config: RLTrainConfig) -> tuple[list[EpisodeResult], dict[str, float]]:
    results: list[EpisodeResult] = []
    steps = 0
    elapsed = 0.0
    for day in days:
        env = MarketMakingEnv(day, config)
        for episode_index, span in enumerate(env.selected_episodes(config.max_eval_episodes_per_day)):
            env.set_eval_context(episode_index)
            env.reset(span)
            done = False
            while not done:
                event_idx = int(env.episode_decisions[env.step_cursor])
                quote_idx = max(event_idx - env.config.latency, env.config.lookback - 1)
                started = perf_counter()
                decision = policy.act(day, quote_idx, env.inventory, env.step_cursor, len(env.episode_decisions))
                elapsed += perf_counter() - started
                _, _, done, _ = env.step(
                    {
                        "ask_price": decision.ask_price,
                        "ask_volume": decision.ask_volume,
                        "bid_price": decision.bid_price,
                        "bid_volume": decision.bid_volume,
                        "spread": decision.spread,
                        "reservation": 0.5 * (decision.ask_price + decision.bid_price),
                    }
                )
                steps += 1
            results.append(env.episode_result(policy.name, episode_index))
    return results, {
        "method": policy.name,
        "inference_steps": float(steps),
        "inference_wall_time_sec": float(elapsed),
        "inference_ms_per_step": float(1000.0 * elapsed / max(steps, 1)),
    }
