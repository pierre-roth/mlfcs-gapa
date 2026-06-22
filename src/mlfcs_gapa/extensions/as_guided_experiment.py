"""Standalone AS-guided PPO extension experiments.

Run with:

    python -m mlfcs_gapa.extensions.as_guided_experiment

This module deliberately does not register with the main `mlfcs-gapa` CLI.
It is an extension sandbox that shares only core data/env/model components.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import typer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.env.gym_env import PaperMarketMakingEnv
from mlfcs_gapa.extensions.as_behavior_cloning import (
    behavior_clone_ppo_policy,
    collect_as_demonstrations,
)
from mlfcs_gapa.extensions.as_guidance import ASGuidanceConfig, make_as_strategy
from mlfcs_gapa.extensions.as_guided_env import ASGuidedMarketMakingEnv
from mlfcs_gapa.paper.constants import PAPER
from mlfcs_gapa.training.ppo import AttnLOBFeatureExtractor


app = typer.Typer(help="AS-guided market-making extension experiments.")

VariantName = Literal["bc_warm_start", "soft_as", "hard_as"]


@dataclass(frozen=True)
class ASGuidedExperimentConfig:
    output_dir: Path
    variant: VariantName
    stock: str = "000001"
    base_price: float = 16.45
    train_events: int = 6_000
    test_events: int = 3_000
    episode_events: int = PAPER.episode_events
    total_timesteps: int = 50_000
    n_envs: int = 4
    seed: int = 1
    soft_penalty: float = 0.1
    hard_window_bias: float = 0.10
    hard_window_spread: float = 0.10
    bc_samples: int = 10_000
    bc_epochs: int = 5
    device: str = "cpu"


def run_synthetic_as_guided_experiment(
    config: ASGuidedExperimentConfig,
) -> tuple[list[dict[str, float | int | str | bool]], list[dict[str, float | int | str]]]:
    """Train and evaluate one AS-guided PPO variant on synthetic data."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = generate_synthetic_lob_day(
        SyntheticLobConfig(
            stock=config.stock,
            day="2019-11-01",
            n_events=config.train_events,
            base_price=config.base_price,
            seed=config.seed,
        )
    )
    test_dataset = generate_synthetic_lob_day(
        SyntheticLobConfig(
            stock=config.stock,
            day="2019-11-04",
            n_events=config.test_events,
            base_price=config.base_price,
            seed=config.seed + 1,
        )
    )
    as_strategy = make_as_strategy(train_dataset, episode_events=config.episode_events)

    train_env = _make_train_env(config, train_dataset, as_strategy)
    model = _make_ppo_model(train_env, config)
    bc_losses: list[dict[str, float | int]] = []
    if config.variant == "bc_warm_start":
        demos = collect_as_demonstrations(
            train_dataset,
            as_strategy=as_strategy,
            n_samples=config.bc_samples,
            episode_events=config.episode_events,
            normalize_actions=True,
            seed=config.seed + 10_000,
        )
        bc_losses = behavior_clone_ppo_policy(
            model,
            demos,
            epochs=config.bc_epochs,
            seed=config.seed + 20_000,
        )
        pl.DataFrame(bc_losses).write_csv(config.output_dir / "bc_losses.csv")

    model.learn(total_timesteps=config.total_timesteps)
    model.save(config.output_dir / f"{config.variant}_ppo_model")

    eval_env = _make_eval_env(config, test_dataset, as_strategy)
    metrics, trades = evaluate_model(model, eval_env, seed=config.seed + 30_000)
    metrics.update(
        {
            "variant": config.variant,
            "stock": config.stock,
            "total_timesteps": config.total_timesteps,
            "bc_epochs": config.bc_epochs if config.variant == "bc_warm_start" else 0,
            "bc_samples": config.bc_samples if config.variant == "bc_warm_start" else 0,
            "soft_penalty": config.soft_penalty if config.variant == "soft_as" else 0.0,
            "as_base_reward": "profit"
            if config.variant in {"soft_as", "hard_as"}
            else "paper_hybrid",
            "hard_window_bias": config.hard_window_bias if config.variant == "hard_as" else 0.0,
            "hard_window_spread": config.hard_window_spread if config.variant == "hard_as" else 0.0,
        }
    )
    for row in trades:
        row["variant"] = config.variant
        row["stock"] = config.stock

    pl.DataFrame([metrics]).write_csv(config.output_dir / "metrics.csv")
    pl.DataFrame(trades).write_parquet(config.output_dir / "trades.parquet")
    pl.DataFrame([_serializable_config(config)]).write_csv(config.output_dir / "config.csv")
    return [metrics], trades


def _make_train_env(config, train_dataset, as_strategy):
    if config.variant == "soft_as":
        guidance = ASGuidanceConfig(
            mode="soft",
            soft_penalty=config.soft_penalty,
            base_reward="profit",
        )

        def make_env(rank: int):
            def _factory():
                return ASGuidedMarketMakingEnv(
                    train_dataset,
                    as_strategy=as_strategy,
                    guidance=guidance,
                    episode_events=config.episode_events,
                    normalize_actions=True,
                    random_episode_starts=True,
                    seed=config.seed + rank,
                )

            return _factory

    elif config.variant == "hard_as":
        guidance = ASGuidanceConfig(
            mode="hard",
            hard_window_bias=config.hard_window_bias,
            hard_window_spread=config.hard_window_spread,
            base_reward="profit",
        )

        def make_env(rank: int):
            def _factory():
                return ASGuidedMarketMakingEnv(
                    train_dataset,
                    as_strategy=as_strategy,
                    guidance=guidance,
                    episode_events=config.episode_events,
                    normalize_actions=True,
                    random_episode_starts=True,
                    seed=config.seed + rank,
                )

            return _factory

    else:

        def make_env(rank: int):
            def _factory():
                return PaperMarketMakingEnv(
                    train_dataset,
                    episode_events=config.episode_events,
                    normalize_actions=True,
                    random_episode_starts=True,
                    seed=config.seed + rank,
                )

            return _factory

    return DummyVecEnv([make_env(rank) for rank in range(max(1, config.n_envs))])


def _serializable_config(config: ASGuidedExperimentConfig) -> dict[str, object]:
    values = asdict(config)
    values["output_dir"] = str(config.output_dir)
    return values


def _make_eval_env(config, test_dataset, as_strategy):
    if config.variant == "hard_as":
        guidance = ASGuidanceConfig(
            mode="hard",
            hard_window_bias=config.hard_window_bias,
            hard_window_spread=config.hard_window_spread,
            base_reward="profit",
        )
        return ASGuidedMarketMakingEnv(
            test_dataset,
            as_strategy=as_strategy,
            guidance=guidance,
            episode_events=min(config.episode_events, config.test_events - 1),
            normalize_actions=True,
            seed=config.seed + 30_000,
        )
    return PaperMarketMakingEnv(
        test_dataset,
        episode_events=min(config.episode_events, config.test_events - 1),
        normalize_actions=True,
        seed=config.seed + 30_000,
    )


def _make_ppo_model(env, config: ASGuidedExperimentConfig):
    n_steps = min(256, max(2, config.episode_events // 2))
    batch_size = max(1, (n_steps * max(1, config.n_envs)) // 4)
    return PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={"features_extractor_class": AttnLOBFeatureExtractor},
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        learning_rate=1e-4,
        gamma=0.99,
        seed=config.seed,
        device=config.device,
        verbose=0,
    )


def evaluate_model(model, env, *, seed: int):
    obs, _ = env.reset(seed=seed)
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
    return metrics, trades


@app.command("run-synthetic")
def run_synthetic_command(
    output_dir: Path = typer.Option(Path("runs/extensions/as_guided"), help="Output directory."),
    variant: VariantName = typer.Option("bc_warm_start", help="Variant to train."),
    stock: str = typer.Option("000001", help="Synthetic stock code."),
    base_price: float = typer.Option(16.45, min=0.01, help="Synthetic base price."),
    train_events: int = typer.Option(6_000, min=200, help="Train synthetic events."),
    test_events: int = typer.Option(3_000, min=200, help="Test synthetic events."),
    episode_events: int = typer.Option(PAPER.episode_events, min=50, help="Episode length."),
    total_timesteps: int = typer.Option(50_000, min=1, help="PPO training timesteps."),
    n_envs: int = typer.Option(4, min=1, help="Parallel PPO environments."),
    seed: int = typer.Option(1, help="Random seed."),
    soft_penalty: float = typer.Option(0.1, min=0.0, help="Soft AS divergence penalty."),
    hard_window_bias: float = typer.Option(0.10, min=0.0, max=1.0, help="Hard AS bias window."),
    hard_window_spread: float = typer.Option(0.10, min=0.0, max=1.0, help="Hard AS spread window."),
    bc_samples: int = typer.Option(10_000, min=1, help="AS demo samples for BC warm start."),
    bc_epochs: int = typer.Option(5, min=1, help="AS BC epochs."),
    device: str = typer.Option("cpu", help="Torch/SB3 device."),
) -> None:
    config = ASGuidedExperimentConfig(
        output_dir=output_dir / variant,
        variant=variant,
        stock=stock,
        base_price=base_price,
        train_events=train_events,
        test_events=test_events,
        episode_events=episode_events,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        seed=seed,
        soft_penalty=soft_penalty,
        hard_window_bias=hard_window_bias,
        hard_window_spread=hard_window_spread,
        bc_samples=bc_samples,
        bc_epochs=bc_epochs,
        device=device,
    )
    metrics, _ = run_synthetic_as_guided_experiment(config)
    typer.echo(f"wrote AS-guided extension run to {config.output_dir}")
    typer.echo(pl.DataFrame(metrics))


if __name__ == "__main__":
    app()
