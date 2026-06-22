"""Latency and AS-teacher diagnostics for saved paper-extension policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path

import numpy as np
import polars as pl
from stable_baselines3 import PPO

from mlfcs_gapa.data.schema import LobDataset
from mlfcs_gapa.extensions.as_guidance import ASGuidanceConfig
from mlfcs_gapa.extensions.as_guided_env import ASGuidedMarketMakingEnv
from mlfcs_gapa.extensions.as_guided_panel import (
    ASCalibrationName,
    ASGuidedPanelConfig,
    DEFAULT_STOCKS,
    SYNTHETIC_STOCK_BASE_PRICES,
    _build_panel,
    _make_as_strategy_for_config,
    _merge_lob_datasets,
)
from mlfcs_gapa.extensions.as_matched_400k_sweep import MethodSpec
from mlfcs_gapa.paper.constants import PAPER, PAPER_PRETRAIN_WINDOWS


STOCKS = tuple(DEFAULT_STOCKS.split(","))
SEED_INDICES = (0, 1, 2)
LATENCIES = (1, 5, 10, 20, 50)


@dataclass(frozen=True)
class DiagnosticSpec:
    method: MethodSpec
    stock: str
    seed_index: int

    @property
    def task_label(self) -> str:
        return f"{self.method.label}/{self.stock}/seed{self.seed_index}"


def method_specs() -> list[MethodSpec]:
    return [
        MethodSpec("paper_cppo", "paper_cppo", soft_penalty=0.0),
        MethodSpec("profit_ppo", "profit_ppo"),
        MethodSpec(
            "soft_as_low_risk_lam_0p10",
            "soft_as",
            soft_penalty=0.10,
            as_calibration="stock_risk_low",
        ),
    ]


def run_specs() -> list[DiagnosticSpec]:
    return [
        DiagnosticSpec(method, stock, seed_index)
        for method in method_specs()
        for stock in STOCKS
        for seed_index in SEED_INDICES
    ]


def run_diagnostics(
    spec: DiagnosticSpec,
    *,
    checkpoint_root: Path,
    output_root: Path,
    device: str,
    latencies: tuple[int, ...],
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    output_dir = output_root / spec.task_label
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        checkpoint_root
        / spec.task_label
        / spec.stock
        / f"{spec.method.label}_seed{spec.seed_index}"
        / "ppo_model.zip"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"missing saved PPO model: {model_path}")

    train_dataset, test_entries = _datasets_for_stock(spec.stock, seed=seed)
    panel_config = _panel_config_for_spec(spec, output_dir=output_dir, seed=seed, device=device)
    as_strategy = _make_as_strategy_for_config(
        train_dataset,
        stock=spec.stock,
        config=panel_config,
    )
    model = PPO.load(model_path, device=device)

    metrics_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for latency in latencies:
        for test_index, (day, dataset) in enumerate(test_entries):
            eval_seed = seed + 1_000 * STOCKS.index(spec.stock) + test_index + 40_000 + latency * 97
            rows = _evaluate_saved_policy(
                spec,
                model=model,
                dataset=dataset,
                as_strategy=as_strategy,
                latency_events=latency,
                seed=eval_seed,
            )
            for episode_metrics, episode_trades in rows:
                episode_metrics.update(
                    {
                        "method": spec.method.label,
                        "variant": spec.method.variant,
                        "stock": spec.stock,
                        "day": day,
                        "train_seed": spec.seed_index,
                        "latency_events": latency,
                        "as_gamma": as_strategy.gamma,
                        "as_kappa": as_strategy.kappa,
                        "as_calibration": spec.method.as_calibration,
                    }
                )
                metrics_rows.append(episode_metrics)
                for trade in episode_trades:
                    tagged = dict(trade)
                    tagged.update(
                        {
                            "method": spec.method.label,
                            "variant": spec.method.variant,
                            "stock": spec.stock,
                            "day": day,
                            "train_seed": spec.seed_index,
                            "episode_id": episode_metrics["episode_id"],
                            "latency_events": latency,
                        }
                    )
                    trade_rows.append(tagged)

    metrics = pl.DataFrame(metrics_rows, infer_schema_length=None)
    trades = _add_teacher_divergence(pl.DataFrame(trade_rows, infer_schema_length=None))
    diagnostics = _diagnostics(trades)

    metrics.write_csv(output_dir / "latency_metrics.csv")
    trades.write_parquet(output_dir / "latency_trades.parquet")
    diagnostics.write_csv(output_dir / "teacher_diagnostics.csv")
    pl.DataFrame(
        [
            {
                "checkpoint_root": str(checkpoint_root),
                "model_path": str(model_path),
                "output_dir": str(output_dir),
                "latencies": ",".join(str(value) for value in latencies),
                **asdict(spec.method),
                "stock": spec.stock,
                "seed_index": spec.seed_index,
            }
        ]
    ).write_csv(output_dir / "diagnostic_config.csv")
    return metrics, diagnostics


def _datasets_for_stock(stock: str, *, seed: int) -> tuple[LobDataset, list[tuple[str, LobDataset]]]:
    stocks = [(stock, SYNTHETIC_STOCK_BASE_PRICES[stock])]
    train_panel = _build_panel(
        stocks=stocks,
        day_indices=range(10),
        events_per_day=10_000,
        seed=seed,
    )
    test_panel = _build_panel(
        stocks=stocks,
        day_indices=range(10, 21),
        events_per_day=10_000,
        seed=seed,
    )
    train_dataset = _merge_lob_datasets(
        [dataset for _, _, dataset in train_panel],
        day="train",
    )
    test_entries = [(day, dataset) for _, day, dataset in test_panel]
    return train_dataset, test_entries


def _panel_config_for_spec(
    spec: DiagnosticSpec,
    *,
    output_dir: Path,
    seed: int,
    device: str,
) -> ASGuidedPanelConfig:
    method = spec.method
    return ASGuidedPanelConfig(
        output_dir=output_dir,
        variant=method.variant,
        label=method.label,
        stocks=(spec.stock,),
        total_timesteps=0,
        agent_seeds=1,
        agent_seed_offset=spec.seed_index,
        n_envs=1,
        seed=seed,
        soft_penalty=method.soft_penalty,
        hard_window_bias=method.hard_window_bias,
        hard_window_spread=method.hard_window_spread,
        as_gamma=method.as_gamma,
        as_kappa=method.as_kappa,
        as_calibration=method.as_calibration,
        eta=PAPER.eta_dampened_pnl,
        zeta=PAPER.zeta_inventory_penalty,
        device=device,
    )


def _guidance_for_diagnostics(method: MethodSpec) -> ASGuidanceConfig:
    if method.variant == "soft_as":
        return ASGuidanceConfig(
            mode="soft",
            soft_penalty=method.soft_penalty,
            base_reward="profit",
        )
    if method.variant == "profit_ppo":
        return ASGuidanceConfig(mode="none", base_reward="profit")
    return ASGuidanceConfig(mode="none", base_reward="paper_hybrid")


def _evaluate_saved_policy(
    spec: DiagnosticSpec,
    *,
    model: PPO,
    dataset: LobDataset,
    as_strategy,
    latency_events: int,
    seed: int,
) -> list[tuple[dict[str, object], list[dict[str, object]]]]:
    episode_events = min(PAPER.episode_events, dataset.orderbook.height - 1)
    env = ASGuidedMarketMakingEnv(
        dataset,
        as_strategy=as_strategy,
        guidance=_guidance_for_diagnostics(spec.method),
        episode_events=episode_events,
        latency_events=latency_events,
        normalize_actions=True,
        eta=PAPER.eta_dampened_pnl,
        zeta=PAPER.zeta_inventory_penalty,
        seed=seed,
    )
    starts = _episode_starts(dataset, episode_events=episode_events, latency_events=latency_events)
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
        tagged_metrics = dict(metrics)
        tagged_metrics.update(
            {
                "episode_id": episode_id,
                "episode_start": episode_start,
                "episode_events": env.episode_events,
            }
        )
        rows.append((tagged_metrics, trades))
    return rows


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


def _add_teacher_divergence(trades: pl.DataFrame) -> pl.DataFrame:
    required = {
        "action_bias",
        "action_spread",
        "teacher_action_bias",
        "teacher_action_spread",
    }
    if not required <= set(trades.columns):
        return trades
    return trades.with_columns(
        (pl.col("action_bias") - pl.col("teacher_action_bias"))
        .abs()
        .alias("teacher_abs_bias_diff"),
        (pl.col("action_spread") - pl.col("teacher_action_spread"))
        .abs()
        .alias("teacher_abs_spread_diff"),
        (
            (pl.col("action_bias") - pl.col("teacher_action_bias")).pow(2)
            + (pl.col("action_spread") - pl.col("teacher_action_spread")).pow(2)
        )
        .sqrt()
        .alias("teacher_l2_diff"),
    )


def _diagnostics(trades: pl.DataFrame) -> pl.DataFrame:
    groups = [
        column
        for column in ("method", "variant", "stock", "train_seed", "day", "episode_id", "latency_events")
        if column in trades.columns
    ]
    expressions = [
        pl.len().alias("log_rows"),
        (pl.col("trade_volume") != 0).sum().alias("fills"),
        pl.col("trade_volume").abs().sum().alias("abs_volume"),
        pl.col("inventory").abs().mean().alias("mean_abs_inventory_log"),
        pl.col("inventory").abs().max().alias("max_abs_inventory"),
        pl.col("value").last().alias("final_value"),
    ]
    for column in (
        "action_bias",
        "action_spread",
        "teacher_action_bias",
        "teacher_action_spread",
        "teacher_abs_bias_diff",
        "teacher_abs_spread_diff",
        "teacher_l2_diff",
        "as_guidance_penalty",
    ):
        if column in trades.columns:
            expressions.append(pl.col(column).mean().alias(f"{column}_mean"))
    return trades.group_by(groups).agg(expressions).sort(groups)


def _parse_latencies(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved policies across latency settings.")
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latencies", default="1,5,10,20,50")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    specs = run_specs()
    if args.list:
        for index, spec in enumerate(specs):
            print(f"{index:03d} {spec.method.label} {spec.stock} seed{spec.seed_index}")
        return
    if not 0 <= args.run_index < len(specs):
        raise SystemExit(f"run-index must be in [0, {len(specs) - 1}]")

    spec = specs[args.run_index]
    print(
        " ".join(
            [
                f"run_index={args.run_index}",
                f"label={spec.method.label}",
                f"stock={spec.stock}",
                f"seed_index={spec.seed_index}",
                f"checkpoint_root={args.checkpoint_root}",
            ]
        ),
        flush=True,
    )
    metrics, diagnostics = run_diagnostics(
        spec,
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_dir,
        device=args.device,
        latencies=_parse_latencies(args.latencies),
        seed=args.seed,
    )
    print(metrics.group_by("method", "stock", "train_seed", "latency_events").agg(pl.len().alias("episodes")), flush=True)
    print(diagnostics.head(), flush=True)


if __name__ == "__main__":
    main()
