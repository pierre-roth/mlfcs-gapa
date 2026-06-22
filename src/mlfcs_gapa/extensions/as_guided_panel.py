"""Panel-scale AS-guided extension experiments.

This runner mirrors the synthetic split used by the paper replication while
remaining separate from the replication CLI. It compares AS-guided C-PPO
variants against the existing paper C-PPO baseline artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import typer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mlfcs_gapa.data.schema import LobDataset
from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.env.baselines import (
    AvellanedaStoikovStrategy,
    estimate_episode_volatility,
    evaluate_quote_strategy,
)
from mlfcs_gapa.env.gym_env import PaperMarketMakingEnv
from mlfcs_gapa.experiments.reports import aggregate_period_table, summarize_paper_table
from mlfcs_gapa.extensions.as_behavior_cloning import (
    behavior_clone_ppo_policy,
    collect_as_demonstrations,
)
from mlfcs_gapa.extensions.as_guidance import ASGuidanceConfig, make_as_strategy
from mlfcs_gapa.extensions.as_guided_env import ASGuidedMarketMakingEnv
from mlfcs_gapa.paper.constants import PAPER, PAPER_PRETRAIN_WINDOWS, PAPER_TRADING_DAYS_201911
from mlfcs_gapa.training.ppo import AttnLOBFeatureExtractor


app = typer.Typer(help="Panel-scale AS-guided extension experiments.")

VariantName = Literal[
    "as_baseline",
    "paper_cppo",
    "bc_warm_start",
    "soft_as",
    "hard_as",
    "profit_ppo",
]
ASCalibrationName = Literal[
    "default",
    "empirical_kappa",
    "stock_specific",
    "spread_kappa",
    "fill_kappa",
    "stock_risk_low",
    "stock_risk_high",
]

SYNTHETIC_STOCK_BASE_PRICES = {
    "000001": 16.45,
    "000858": 130.0,
    "002415": 35.0,
}
DEFAULT_STOCKS = "000001,000858,002415"
PPO_LOG_STD_INIT = -2.0


@dataclass(frozen=True)
class ASGuidedPanelConfig:
    output_dir: Path
    variant: VariantName
    label: str
    stocks: tuple[str, ...] = ("000001", "000858", "002415")
    train_days: int = 10
    test_days: int = 11
    events_per_day: int = 10_000
    episode_events: int = PAPER.episode_events
    total_timesteps: int = 200_000
    agent_seeds: int = 3
    agent_seed_offset: int = 0
    n_envs: int = 8
    seed: int = 101
    soft_penalty: float = 0.1
    hard_window_bias: float = 0.10
    hard_window_spread: float = 0.10
    bias_weight: float = 1.0
    spread_weight: float = 1.0
    penalty_norm: str = "l2"
    penalty_space: str = "action"
    soft_penalty_end: float | None = None
    penalty_schedule: str = "constant"
    huber_delta: float = 0.10
    adaptive_target: float = 0.15
    as_gamma: float = 1.0
    as_kappa: float = 100.0
    as_calibration: ASCalibrationName = "default"
    eta: float = PAPER.eta_dampened_pnl
    zeta: float = PAPER.zeta_inventory_penalty
    bc_samples: int = 20_000
    bc_epochs: int = 3
    bc_learning_rate: float = 1e-4
    encoder_checkpoint: str | None = None
    freeze_encoder: bool = False
    device: str = "cuda"


def run_as_guided_panel(
    config: ASGuidedPanelConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run one panel-scale extension configuration."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    stocks = [(stock, SYNTHETIC_STOCK_BASE_PRICES[stock]) for stock in config.stocks]
    train_panel = _build_panel(
        stocks=stocks,
        day_indices=range(config.train_days),
        events_per_day=config.events_per_day,
        seed=config.seed,
    )
    test_panel = _build_panel(
        stocks=stocks,
        day_indices=range(config.train_days, config.train_days + config.test_days),
        events_per_day=config.events_per_day,
        seed=config.seed,
    )

    metrics_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for stock_index, (stock, _) in enumerate(stocks):
        train_dataset = _merge_lob_datasets(
            [dataset for panel_stock, _, dataset in train_panel if panel_stock == stock],
            day="train",
        )
        test_entries = [(day, dataset) for panel_stock, day, dataset in test_panel if panel_stock == stock]
        as_strategy = _make_as_strategy_for_config(
            train_dataset,
            stock=stock,
            config=config,
        )
        guidance = _guidance_for_config(config)

        for seed_index in range(
            config.agent_seed_offset,
            config.agent_seed_offset + config.agent_seeds,
        ):
            model_seed = config.seed + 30_000 + 199 * seed_index + stock_index
            model = _train_variant_model(
                config,
                train_dataset=train_dataset,
                as_strategy=as_strategy,
                output_dir=config.output_dir / stock / f"{config.label}_seed{seed_index}",
                seed=model_seed,
            )
            for test_index, (day, dataset) in enumerate(test_entries):
                eval_seed = config.seed + 1_000 * stock_index + test_index + 40_000
                rows = _evaluate_variant_on_dataset(
                    config,
                    model=model,
                    test_dataset=dataset,
                    as_strategy=as_strategy,
                    seed=eval_seed,
                )
                for episode_metrics, episode_trades in rows:
                    episode_metrics.update(
                        {
                            "method": config.label,
                            "variant": config.variant,
                            "stock": stock,
                            "day": day,
                            "train_seed": seed_index,
                            "total_timesteps": config.total_timesteps,
                            "soft_penalty": config.soft_penalty if config.variant == "soft_as" else 0.0,
                            "as_base_reward": guidance.base_reward,
                            "bias_weight": guidance.bias_weight,
                            "spread_weight": guidance.spread_weight,
                            "penalty_norm": guidance.penalty_norm,
                            "penalty_space": guidance.penalty_space,
                            "soft_penalty_end": guidance.soft_penalty_end,
                            "penalty_schedule": guidance.penalty_schedule,
                            "as_gamma": as_strategy.gamma,
                            "as_kappa": as_strategy.kappa,
                            "as_calibration": config.as_calibration,
                            "eta": config.eta,
                            "zeta": config.zeta,
                            "hard_window_bias": (
                                config.hard_window_bias if config.variant == "hard_as" else 0.0
                            ),
                            "hard_window_spread": (
                                config.hard_window_spread if config.variant == "hard_as" else 0.0
                            ),
                            "bc_samples": config.bc_samples
                            if config.variant == "bc_warm_start"
                            else 0,
                            "bc_epochs": config.bc_epochs if config.variant == "bc_warm_start" else 0,
                            "encoder_checkpoint": config.encoder_checkpoint,
                            "freeze_encoder": config.freeze_encoder,
                        }
                    )
                    metrics_rows.append(episode_metrics)
                    for trade in episode_trades:
                        tagged = dict(trade)
                        tagged.update(
                            {
                                "method": config.label,
                                "variant": config.variant,
                                "stock": stock,
                                "day": day,
                                "train_seed": seed_index,
                                "episode_id": episode_metrics["episode_id"],
                            }
                        )
                        trade_rows.append(tagged)

    metrics = pl.DataFrame(metrics_rows, infer_schema_length=None)
    trades = pl.DataFrame(trade_rows, infer_schema_length=None)
    metrics.write_csv(config.output_dir / "extension_metrics.csv")
    trades.write_parquet(config.output_dir / "extension_trades.parquet")
    summarize_paper_table(metrics).write_csv(config.output_dir / "extension_summary.csv")
    aggregate_period_table(metrics).write_csv(config.output_dir / "extension_paper_table.csv")
    _trade_diagnostics(trades).write_csv(config.output_dir / "extension_trade_diagnostics.csv")
    pl.DataFrame([_serializable_config(config)]).write_csv(config.output_dir / "extension_config.csv")
    return metrics, trades


def run_as_baseline_panel(
    config: ASGuidedPanelConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluate fitted AS directly on the same synthetic panel split."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    stocks = [(stock, SYNTHETIC_STOCK_BASE_PRICES[stock]) for stock in config.stocks]
    train_panel = _build_panel(
        stocks=stocks,
        day_indices=range(config.train_days),
        events_per_day=config.events_per_day,
        seed=config.seed,
    )
    test_panel = _build_panel(
        stocks=stocks,
        day_indices=range(config.train_days, config.train_days + config.test_days),
        events_per_day=config.events_per_day,
        seed=config.seed,
    )

    metrics_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for stock_index, (stock, _) in enumerate(stocks):
        train_dataset = _merge_lob_datasets(
            [dataset for panel_stock, _, dataset in train_panel if panel_stock == stock],
            day="train",
        )
        test_entries = [(day, dataset) for panel_stock, day, dataset in test_panel if panel_stock == stock]
        as_strategy = _make_as_strategy_for_config(train_dataset, stock=stock, config=config)

        for test_index, (day, dataset) in enumerate(test_entries):
            episode_events = min(config.episode_events, dataset.orderbook.height - 1)
            starts = _episode_starts(dataset, episode_events=episode_events, latency_events=1)
            for episode_id, episode_start in enumerate(starts):
                eval_seed = config.seed + 1_000 * stock_index + test_index + 40_000
                episode_metrics, episode_trades = evaluate_quote_strategy(
                    dataset,
                    as_strategy,
                    episode_start=episode_start,
                    episode_events=episode_events,
                    latency_events=1,
                    seed=eval_seed + episode_id,
                )
                episode_metrics.update(
                    {
                        "method": config.label,
                        "variant": "as_baseline",
                        "stock": stock,
                        "day": day,
                        "train_seed": -1,
                        "episode_id": episode_id,
                        "episode_start": episode_start,
                        "episode_events": episode_events,
                        "latency_events": 1,
                        "total_timesteps": 0,
                        "soft_penalty": 0.0,
                        "as_base_reward": "direct_quote",
                        "bias_weight": 0.0,
                        "spread_weight": 0.0,
                        "penalty_norm": "none",
                        "penalty_space": "none",
                        "soft_penalty_end": None,
                        "penalty_schedule": "none",
                        "as_gamma": as_strategy.gamma,
                        "as_kappa": as_strategy.kappa,
                        "as_calibration": config.as_calibration,
                        "eta": 0.0,
                        "zeta": 0.0,
                        "hard_window_bias": 0.0,
                        "hard_window_spread": 0.0,
                        "bc_samples": 0,
                        "bc_epochs": 0,
                        "encoder_checkpoint": config.encoder_checkpoint,
                        "freeze_encoder": config.freeze_encoder,
                    }
                )
                metrics_rows.append(episode_metrics)
                for trade in episode_trades:
                    tagged = dict(trade)
                    tagged.update(
                        {
                            "method": config.label,
                            "variant": "as_baseline",
                            "stock": stock,
                            "day": day,
                            "train_seed": -1,
                            "episode_id": episode_id,
                        }
                    )
                    trade_rows.append(tagged)

    metrics = pl.DataFrame(metrics_rows, infer_schema_length=None)
    trades = pl.DataFrame(trade_rows, infer_schema_length=None)
    metrics.write_csv(config.output_dir / "extension_metrics.csv")
    trades.write_parquet(config.output_dir / "extension_trades.parquet")
    summarize_paper_table(metrics).write_csv(config.output_dir / "extension_summary.csv")
    aggregate_period_table(metrics).write_csv(config.output_dir / "extension_paper_table.csv")
    _trade_diagnostics(trades).write_csv(config.output_dir / "extension_trade_diagnostics.csv")
    pl.DataFrame([_serializable_config(config)]).write_csv(config.output_dir / "extension_config.csv")
    return metrics, trades


def _train_variant_model(config, *, train_dataset, as_strategy, output_dir, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    env = _make_train_env(config, train_dataset=train_dataset, as_strategy=as_strategy, seed=seed)
    model = _make_ppo_model(env, config, seed=seed)
    if config.variant == "bc_warm_start":
        demos = collect_as_demonstrations(
            train_dataset,
            as_strategy=as_strategy,
            n_samples=config.bc_samples,
            episode_events=config.episode_events,
            normalize_actions=True,
            seed=seed + 10_000,
        )
        losses = behavior_clone_ppo_policy(
            model,
            demos,
            epochs=config.bc_epochs,
            learning_rate=config.bc_learning_rate,
            seed=seed + 20_000,
        )
        pl.DataFrame(losses).write_csv(output_dir / "bc_losses.csv")
    model.learn(total_timesteps=config.total_timesteps)
    model.save(output_dir / "ppo_model")
    return model


def _make_train_env(config, *, train_dataset, as_strategy, seed):
    guidance = _guidance_for_config(config)

    def make_env(rank: int):
        def _factory():
            if config.variant in {"bc_warm_start", "soft_as", "hard_as", "profit_ppo"}:
                return ASGuidedMarketMakingEnv(
                    train_dataset,
                    as_strategy=as_strategy,
                    guidance=guidance,
                    episode_events=config.episode_events,
                    latency_events=1,
                    normalize_actions=True,
                    random_episode_starts=True,
                    eta=config.eta,
                    zeta=config.zeta,
                    seed=seed + rank,
                )
            return PaperMarketMakingEnv(
                train_dataset,
                episode_events=config.episode_events,
                latency_events=1,
                normalize_actions=True,
                random_episode_starts=True,
                eta=config.eta,
                zeta=config.zeta,
                seed=seed + rank,
            )

        return _factory

    return DummyVecEnv([make_env(rank) for rank in range(max(1, config.n_envs))])


def _make_eval_env(config, *, test_dataset, as_strategy, seed):
    guidance = _guidance_for_config(config)
    episode_events = min(config.episode_events, test_dataset.orderbook.height - 1)
    if config.variant in {"bc_warm_start", "soft_as", "hard_as", "profit_ppo"}:
        return ASGuidedMarketMakingEnv(
            test_dataset,
            as_strategy=as_strategy,
            guidance=guidance,
            episode_events=episode_events,
            latency_events=1,
            normalize_actions=True,
            eta=config.eta,
            zeta=config.zeta,
            seed=seed,
        )
    return PaperMarketMakingEnv(
        test_dataset,
        episode_events=episode_events,
        latency_events=1,
        normalize_actions=True,
        eta=config.eta,
        zeta=config.zeta,
        seed=seed,
    )


def _make_ppo_model(env, config: ASGuidedPanelConfig, *, seed: int):
    n_steps = min(256, max(2, config.episode_events // 2))
    batch_size = max(1, (n_steps * max(1, config.n_envs)) // 4)
    return PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": AttnLOBFeatureExtractor,
            "features_extractor_kwargs": {
                "encoder_checkpoint": config.encoder_checkpoint,
                "freeze_encoder": config.freeze_encoder,
            },
            "log_std_init": PPO_LOG_STD_INIT,
        },
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        learning_rate=1e-4,
        gamma=0.99,
        seed=seed,
        device=config.device,
        verbose=0,
    )


def _evaluate_variant_on_dataset(config, *, model, test_dataset, as_strategy, seed):
    env = _make_eval_env(config, test_dataset=test_dataset, as_strategy=as_strategy, seed=seed)
    starts = _episode_starts(test_dataset, episode_events=env.episode_events, latency_events=1)
    rows = []
    for episode_id, episode_start in enumerate(starts):
        obs, _ = env.reset(seed=seed + episode_id, options={"episode_start": episode_start})
        done = False
        info: dict[str, object] = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        metrics = info.get("metrics", {})
        trades = info.get("trade_log", [])
        if not isinstance(metrics, dict) or not isinstance(trades, list):
            raise RuntimeError("evaluation did not return terminal metrics")
        metrics = dict(metrics)
        metrics.update(
            {
                "episode_id": episode_id,
                "episode_start": episode_start,
                "episode_events": env.episode_events,
                "latency_events": 1,
            }
        )
        rows.append((metrics, trades))
    return rows


def _guidance_for_config(config: ASGuidedPanelConfig) -> ASGuidanceConfig:
    if config.variant == "soft_as":
        return ASGuidanceConfig(
            mode="soft",
            soft_penalty=config.soft_penalty,
            base_reward="profit",
            bias_weight=config.bias_weight,
            spread_weight=config.spread_weight,
            penalty_norm=config.penalty_norm,  # type: ignore[arg-type]
            penalty_space=config.penalty_space,  # type: ignore[arg-type]
            soft_penalty_end=config.soft_penalty_end,
            penalty_schedule=config.penalty_schedule,  # type: ignore[arg-type]
            huber_delta=config.huber_delta,
            adaptive_target=config.adaptive_target,
        )
    if config.variant == "hard_as":
        return ASGuidanceConfig(
            mode="hard",
            hard_window_bias=config.hard_window_bias,
            hard_window_spread=config.hard_window_spread,
            base_reward="profit",
        )
    if config.variant == "profit_ppo":
        return ASGuidanceConfig(mode="none", base_reward="profit")
    if config.variant == "bc_warm_start":
        return ASGuidanceConfig(mode="none", base_reward="profit")
    return ASGuidanceConfig(mode="none", base_reward="paper_hybrid")


def _make_as_strategy_for_config(
    dataset: LobDataset,
    *,
    stock: str,
    config: ASGuidedPanelConfig,
) -> AvellanedaStoikovStrategy:
    gamma = config.as_gamma
    kappa = config.as_kappa
    if config.as_calibration == "empirical_kappa":
        kappa = _estimate_kappa_from_l1_spread(dataset)
    elif config.as_calibration == "spread_kappa":
        kappa = _estimate_kappa_from_l1_spread(dataset)
    elif config.as_calibration == "fill_kappa":
        kappa = _estimate_kappa_from_fill_decay(dataset)
        gamma = _stock_specific_gamma(dataset, stock=stock)
    elif config.as_calibration == "stock_specific":
        kappa = _estimate_kappa_from_l1_spread(dataset)
        gamma = _stock_specific_gamma(dataset, stock=stock)
    elif config.as_calibration == "stock_risk_low":
        kappa = _estimate_kappa_from_l1_spread(dataset)
        gamma = _scale_gamma(_stock_specific_gamma(dataset, stock=stock), 0.5)
    elif config.as_calibration == "stock_risk_high":
        kappa = _estimate_kappa_from_l1_spread(dataset)
        gamma = _scale_gamma(_stock_specific_gamma(dataset, stock=stock), 2.0)
    elif config.as_calibration != "default":
        raise ValueError("unknown AS calibration")
    return make_as_strategy(
        dataset,
        episode_events=config.episode_events,
        gamma=gamma,
        kappa=kappa,
    )


def _estimate_kappa_from_l1_spread(dataset: LobDataset) -> float:
    spread = (
        dataset.orderbook["ask1_price"].to_numpy()
        - dataset.orderbook["bid1_price"].to_numpy()
    )
    mean_half_spread = max(float(spread.mean()) / 2.0, 1e-4)
    return float(min(300.0, max(10.0, 1.0 / mean_half_spread)))


def _estimate_kappa_from_fill_decay(dataset: LobDataset) -> float:
    ask1 = dataset.orderbook["ask1_price"].to_numpy().astype(np.float64)
    bid1 = dataset.orderbook["bid1_price"].to_numpy().astype(np.float64)
    trade_max = dataset.trades["trade_price_max"].to_numpy().astype(np.float64)
    trade_min = dataset.trades["trade_price_min"].to_numpy().astype(np.float64)
    trade_max_volume = dataset.trades["trade_price_max_volume"].to_numpy().astype(np.int64)
    trade_min_volume = dataset.trades["trade_price_min_volume"].to_numpy().astype(np.int64)

    tick_size = 0.01
    distances: list[float] = []
    probabilities: list[float] = []
    for ticks in range(0, 7):
        distance = ticks * tick_size
        ask_quote = ask1 + distance
        bid_quote = bid1 - distance
        ask_fill = (trade_max_volume > 0) & (trade_max > ask_quote)
        bid_fill = (trade_min_volume > 0) & (trade_min < bid_quote)
        probability = float((ask_fill.sum() + bid_fill.sum()) / max(1, 2 * len(ask_fill)))
        if probability > 1e-5:
            distances.append(distance)
            probabilities.append(probability)

    if len(distances) < 3:
        return _estimate_kappa_from_l1_spread(dataset)
    slope, _ = np.polyfit(
        np.asarray(distances, dtype=np.float64),
        np.log(np.asarray(probabilities, dtype=np.float64)),
        deg=1,
    )
    kappa = -float(slope)
    if not np.isfinite(kappa) or kappa <= 0.0:
        return _estimate_kappa_from_l1_spread(dataset)
    return float(min(300.0, max(10.0, kappa)))


def _stock_specific_gamma(dataset: LobDataset, *, stock: str) -> float:
    del stock
    sigma = max(estimate_episode_volatility(dataset), 1e-6)
    return float(min(2.0, max(0.05, 0.0025 / (sigma * sigma))))


def _scale_gamma(gamma: float, multiplier: float) -> float:
    return float(min(4.0, max(0.025, gamma * multiplier)))


def _build_panel(*, stocks, day_indices, events_per_day: int, seed: int):
    panel = []
    for stock, base_price in stocks:
        stock_index = _stock_index(stock)
        for day_index in day_indices:
            day = PAPER_TRADING_DAYS_201911[day_index]
            dataset = generate_synthetic_lob_day(
                SyntheticLobConfig(
                    stock=stock,
                    day=day,
                    n_events=events_per_day,
                    base_price=base_price,
                    seed=seed + 1_000 * stock_index + day_index,
                )
            )
            panel.append((stock, day, _filter_stable_windows(dataset)))
    return panel


def _stock_index(stock: str) -> int:
    stock_order = tuple(SYNTHETIC_STOCK_BASE_PRICES)
    try:
        return stock_order.index(stock)
    except ValueError as exc:
        raise ValueError(f"unknown synthetic stock code: {stock}") from exc


def _filter_stable_windows(dataset: LobDataset) -> LobDataset:
    windows = [
        (time.fromisoformat(start), time.fromisoformat(end))
        for start, end in PAPER_PRETRAIN_WINDOWS
    ]
    mask = [
        any(start <= timestamp.time() <= end for start, end in windows)
        for timestamp in dataset.orderbook["timestamp"].to_list()
    ]
    return LobDataset(
        stock=dataset.stock,
        day=dataset.day,
        orderbook=dataset.orderbook.filter(mask),
        messages=dataset.messages.filter(mask),
        trades=dataset.trades.filter(mask),
    )


def _merge_lob_datasets(datasets: list[LobDataset], *, day: str) -> LobDataset:
    if not datasets:
        raise ValueError("expected at least one dataset")
    stock = datasets[0].stock
    return LobDataset(
        stock=stock,
        day=day,
        orderbook=pl.concat([dataset.orderbook for dataset in datasets], how="vertical"),
        messages=pl.concat([dataset.messages for dataset in datasets], how="vertical"),
        trades=pl.concat([dataset.trades for dataset in datasets], how="vertical"),
    )


def _episode_starts(dataset: LobDataset, *, episode_events: int, latency_events: int) -> list[int]:
    if episode_events <= PAPER.window_length + latency_events:
        return []
    windows = [
        (time.fromisoformat(start), time.fromisoformat(end))
        for start, end in PAPER_PRETRAIN_WINDOWS
    ]
    timestamps = dataset.orderbook["timestamp"].to_list()
    starts: list[int] = []
    for window_start, window_end in windows:
        indices = [
            index
            for index, timestamp in enumerate(timestamps)
            if window_start <= timestamp.time() <= window_end
        ]
        if not indices:
            continue
        start = indices[0]
        last_exclusive = indices[-1] + 1
        while start + episode_events <= last_exclusive:
            starts.append(start)
            start += episode_events
    if starts:
        return starts
    if dataset.orderbook.height > episode_events:
        return [0]
    return []


def _trade_diagnostics(trades: pl.DataFrame) -> pl.DataFrame:
    groups = [column for column in ("method", "variant", "stock", "train_seed", "episode_id") if column in trades.columns]
    return (
        trades.group_by(groups)
        .agg(
            pl.len().alias("log_rows"),
            (pl.col("trade_volume") != 0).sum().alias("fills"),
            pl.col("trade_volume").abs().sum().alias("abs_volume"),
            pl.col("inventory").abs().mean().alias("mean_abs_inventory_log"),
            pl.col("inventory").abs().max().alias("max_abs_inventory"),
            pl.col("value").last().alias("final_value"),
            *[
                pl.col(column).mean().alias(f"{column}_mean")
                for column in (
                    "action_bias",
                    "action_spread",
                    "raw_action_bias",
                    "raw_action_spread",
                    "teacher_action_bias",
                    "teacher_action_spread",
                    "as_guidance_penalty",
                    "as_guidance_penalty_scale",
                )
                if column in trades.columns
            ],
        )
        .sort(groups)
    )


def _serializable_config(config: ASGuidedPanelConfig) -> dict[str, object]:
    values = asdict(config)
    values["output_dir"] = str(config.output_dir)
    values["stocks"] = ",".join(config.stocks)
    return values


def _parse_stocks(stocks: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in stocks.split(",") if part.strip())
    missing = [stock for stock in parsed if stock not in SYNTHETIC_STOCK_BASE_PRICES]
    if missing:
        raise typer.BadParameter(f"unknown synthetic stock code(s): {missing}")
    return parsed


@app.command("run")
def run_command(
    output_dir: Path = typer.Option(Path("runs/extensions/as_guided_panel")),
    variant: VariantName = typer.Option("bc_warm_start"),
    label: str | None = typer.Option(None, help="Label used in output tables."),
    stocks: str = typer.Option(DEFAULT_STOCKS),
    train_days: int = typer.Option(10, min=1, max=20),
    test_days: int = typer.Option(11, min=1, max=20),
    events_per_day: int = typer.Option(10_000, min=1_000),
    episode_events: int = typer.Option(PAPER.episode_events, min=100),
    total_timesteps: int = typer.Option(200_000, min=1),
    agent_seeds: int = typer.Option(3, min=1),
    n_envs: int = typer.Option(8, min=1),
    seed: int = typer.Option(101),
    soft_penalty: float = typer.Option(0.1, min=0.0),
    hard_window_bias: float = typer.Option(0.10, min=0.0, max=1.0),
    hard_window_spread: float = typer.Option(0.10, min=0.0, max=1.0),
    bc_samples: int = typer.Option(20_000, min=1),
    bc_epochs: int = typer.Option(3, min=1),
    bc_learning_rate: float = typer.Option(1e-4, min=1e-8),
    encoder_checkpoint: str | None = typer.Option(None),
    freeze_encoder: bool = typer.Option(False, "--freeze-encoder/--fine-tune-encoder"),
    device: str = typer.Option("cuda"),
) -> None:
    run_label = label or variant
    config = ASGuidedPanelConfig(
        output_dir=output_dir / run_label,
        variant=variant,
        label=run_label,
        stocks=_parse_stocks(stocks),
        train_days=train_days,
        test_days=test_days,
        events_per_day=events_per_day,
        episode_events=episode_events,
        total_timesteps=total_timesteps,
        agent_seeds=agent_seeds,
        n_envs=n_envs,
        seed=seed,
        soft_penalty=soft_penalty,
        hard_window_bias=hard_window_bias,
        hard_window_spread=hard_window_spread,
        bc_samples=bc_samples,
        bc_epochs=bc_epochs,
        bc_learning_rate=bc_learning_rate,
        encoder_checkpoint=encoder_checkpoint,
        freeze_encoder=freeze_encoder,
        device=device,
    )
    metrics, _ = run_as_guided_panel(config)
    typer.echo(f"wrote AS-guided panel run to {config.output_dir}")
    typer.echo(aggregate_period_table(metrics))


if __name__ == "__main__":
    app()
