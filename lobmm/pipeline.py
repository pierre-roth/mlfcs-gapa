from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd
import torch

from .baselines import AvellanedaStoikovPolicy, BaselinePolicy, FixedLevelPolicy, QuoteDecision, RandomPolicy
from .config import ExperimentConfig, RLTrainConfig
from .data import DayData, apply_lob_normalizer, discover_days, fit_lob_normalizer, load_day_data, split_days
from .env import MarketMakingEnv
from .metrics import EpisodeResult, sharpe
from .utils import ensure_dir, save_json, set_seed, timestamped_name


def prepare_run(config: ExperimentConfig) -> Path:
    set_seed(config.seed)
    if not config.run_name:
        config.run_name = timestamped_name(config.mode)
    out = config.output_dir()
    ensure_dir(out)
    save_json(out / "config.json", config)
    return out


def load_symbol_splits(config: ExperimentConfig, symbol: str) -> dict[str, list[DayData]]:
    days = discover_days(config.data_dir, symbol)
    train_days, val_days, test_days = split_days(days, config.train_days, config.val_days, config.test_days)
    splits = {"train": train_days, "val": val_days, "test": test_days}
    loaded: dict[str, list[DayData]] = {}
    for split_name, split_list in splits.items():
        loaded[split_name] = [load_day_data(symbol, day, config) for day in split_list]
    mean, std = fit_lob_normalizer(loaded["train"])
    for split_days_data in loaded.values():
        for day in split_days_data:
            apply_lob_normalizer(day, mean, std)
    return loaded


def save_episode_results(path: str | Path, results: Iterable[EpisodeResult]) -> pd.DataFrame:
    frame = pd.DataFrame([result.to_dict() for result in results])
    ensure_dir(Path(path).parent)
    frame.to_csv(path, index=False)
    return frame


def summarize_results(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "episodes": int(len(frame)),
        "pnl_mean": float(frame["pnl"].mean()) if not frame.empty else 0.0,
        "nd_pnl_mean": float(frame["nd_pnl"].mean()) if not frame.empty else 0.0,
        "pnl_map_mean": float(frame["pnl_map"].mean()) if not frame.empty else 0.0,
        "profit_ratio_mean": float(frame["profit_ratio"].mean()) if not frame.empty else 0.0,
        "sharpe": sharpe(frame["pnl"].tolist()) if not frame.empty else 0.0,
    }


def evaluate_baseline_policy(
    policy: BaselinePolicy,
    days: list[DayData],
    config: RLTrainConfig,
    latency: int | None = None,
) -> tuple[list[EpisodeResult], dict[str, float]]:
    results: list[EpisodeResult] = []
    latency_value = latency if latency is not None else config.latency
    inference_steps = 0
    inference_elapsed = 0.0
    for day in days:
        eval_cfg = RLTrainConfig(**asdict(config))
        eval_cfg.latency = latency_value
        env = MarketMakingEnv(day, eval_cfg, state_mode="full", reward_mode=config.reward_mode)
        for episode_index, span in enumerate(env.available_episodes()[: config.max_eval_episodes_per_day]):
            env.reset(span)
            done = False
            while not done:
                event_idx = int(env.episode_decisions[env.step_cursor])
                quote_idx = max(event_idx - env.config.latency, env.config.lookback - 1)
                started = perf_counter()
                decision = policy.act(day, quote_idx, env.inventory, env.step_cursor, len(env.episode_decisions))
                inference_elapsed += perf_counter() - started
                inference_steps += 1
                action = {
                    "ask_price": decision.ask_price,
                    "ask_volume": decision.ask_volume,
                    "bid_price": decision.bid_price,
                    "bid_volume": decision.bid_volume,
                    "spread": decision.spread,
                }
                _, _, done, _ = env.step(action)
            results.append(env.episode_result(policy.name, episode_index, latency=latency_value))
    return results, {
        "method": policy.name,
        "inference_steps": float(inference_steps),
        "inference_wall_time_sec": float(inference_elapsed),
        "inference_ms_per_step": float(1000.0 * inference_elapsed / max(inference_steps, 1)),
    }


def standard_baselines(config: ExperimentConfig) -> list[BaselinePolicy]:
    return [
        AvellanedaStoikovPolicy(config),
        RandomPolicy(config),
        FixedLevelPolicy(config, 1),
        FixedLevelPolicy(config, 2),
        FixedLevelPolicy(config, 3),
    ]
