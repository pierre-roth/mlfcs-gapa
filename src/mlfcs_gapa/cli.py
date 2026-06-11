"""Command-line entrypoints for replication utilities."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from time import perf_counter

import polars as pl
import torch
import typer
from rich.console import Console

from mlfcs_gapa.data.features import normalize_lob_window
from mlfcs_gapa.data.io import write_lob_dataset
from mlfcs_gapa.data.pretraining import build_pretrain_arrays
from mlfcs_gapa.data.schema import LobDataset, lob_columns
from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.env.baselines import (
    AvellanedaStoikovStrategy,
    FixedLevelStrategy,
    RandomLevelStrategy,
    estimate_event_volatility,
    evaluate_quote_strategy,
)
from mlfcs_gapa.env.discrete_env import PaperDiscreteMarketMakingEnv
from mlfcs_gapa.env.tabular_rl import (
    BestBidAskActionSpace,
    InventoryTimeEncoder,
    LobRlEncoder,
    OffsetActionSpace,
    QLearningConfig,
    train_and_evaluate_tabular_baseline,
    train_tabular_q_strategy,
)
from mlfcs_gapa.env.gym_env import PaperMarketMakingEnv
from mlfcs_gapa.experiments.figures import (
    plot_attention_heatmap,
    plot_decision_trace,
    plot_latency_figure,
)
from mlfcs_gapa.experiments.reports import summarize_paper_table
from mlfcs_gapa.experiments.tracking import (
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    WandbTracker,
    init_wandb_run,
    wandb_run,
)
from mlfcs_gapa.models.attn_lob import AttnLOBClassifier
from mlfcs_gapa.models.pretrain_models import (
    count_encoder_parameters,
    count_parameters,
    make_pretrain_model,
    paper_reported_parameter_count,
    pretrain_input_shape,
)
from mlfcs_gapa.paper.constants import PAPER, PAPER_PRETRAIN_WINDOWS, PAPER_TRADING_DAYS_201911
from mlfcs_gapa.training.dueling_dqn import (
    DuelingDQNConfig,
    evaluate_dueling_dqn,
    save_dueling_dqn,
    train_dueling_dqn,
)
from mlfcs_gapa.training.ppo import AttnLOBFeatureExtractor
from mlfcs_gapa.training.pretrain import train_lob_classifier

app = typer.Typer(no_args_is_help=True)
console = Console()

SYNTHETIC_STOCK_BASE_PRICES: dict[str, float] = {
    "000001": 16.45,
    "000858": 130.00,
    "002415": 35.00,
}
PRETRAIN_MODEL_NAMES: tuple[str, ...] = ("FC-LOB", "Conv-LOB", "DeepLOB", "Attn-LOB")
PAPER_METHOD_ORDER: tuple[str, ...] = (
    "C-PPO",
    "D-DQN",
    "Inv-RL",
    "LOB-RL",
    "AS",
    "Random",
    "Fixed_1",
    "Fixed_2",
    "Fixed_3",
)
PAPER_TRAIN_DAYS = 10
PAPER_TEST_DAYS = len(PAPER_TRADING_DAYS_201911) - PAPER_TRAIN_DAYS
FULL_REPLICATION_EVENTS_PER_DAY = PAPER.episode_events * 3
FULL_REPLICATION_AGENT_TIMESTEPS = PAPER.episode_events * PAPER_TRAIN_DAYS * 5
FULL_REPLICATION_PPO_LOG_STD_INIT = -2.0


@app.callback()
def main() -> None:
    """Utilities for the paper-faithful LOB market-making replication."""


@app.command("generate-synthetic")
def generate_synthetic(
    output_dir: Path = typer.Option(Path("data/synthetic"), help="Output root for Parquet files."),
    stock: str = typer.Option("000001", help="Synthetic stock code."),
    stocks: str | None = typer.Option(
        None, help="Comma-separated synthetic stock codes. Overrides --stock when provided."
    ),
    days: int = typer.Option(1, min=1, max=len(PAPER_TRADING_DAYS_201911), help="Number of days."),
    events_per_day: int = typer.Option(6_000, min=100, help="Events generated per day."),
    base_price: float = typer.Option(16.45, min=0.01, help="Starting stock price."),
    base_prices: str | None = typer.Option(
        None, help="Comma-separated base prices matching --stocks."
    ),
    seed: int = typer.Option(1, help="Base random seed."),
) -> None:
    """Generate paper-shaped synthetic LOB data.

    The output is source-separated under ``data/synthetic`` by default and uses
    the canonical schema consumed by the replication pipeline.
    """

    stock_specs = _parse_synthetic_stock_specs(stocks or stock, base_prices, fallback=base_price)
    for stock_index, (stock_code, stock_base_price) in enumerate(stock_specs):
        for day_index, day in enumerate(PAPER_TRADING_DAYS_201911[:days]):
            dataset = _generate_configured_synthetic_day(
                stock=stock_code,
                day=day,
                events=events_per_day,
                base_price=stock_base_price,
                seed=seed + 1_000 * stock_index + day_index,
            )
            written = write_lob_dataset(dataset, output_dir)
            console.print(
                f"[green]wrote[/green] {written} "
                f"({dataset.orderbook.height:,} events, {len(dataset.orderbook.columns)} orderbook columns)"
            )

@app.command("run-synthetic-baselines")
def run_synthetic_baselines(
    output_dir: Path = typer.Option(Path("runs/synthetic-baselines"), help="Output directory."),
    days: int = typer.Option(1, min=1, max=len(PAPER_TRADING_DAYS_201911), help="Synthetic days."),
    events_per_day: int = typer.Option(3_000, min=200, help="Events generated per synthetic day."),
    episode_events: int = typer.Option(2_000, min=100, help="Events evaluated per episode."),
    tabular_episodes: int = typer.Option(20, min=1, help="Q-learning episodes for Inv-RL/LOB-RL."),
    seed: int = typer.Option(1, help="Base random seed."),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Log this run to W&B."),
    wandb_entity: str = typer.Option(DEFAULT_WANDB_ENTITY, help="W&B entity/team."),
    wandb_project: str = typer.Option(DEFAULT_WANDB_PROJECT, help="W&B project."),
    wandb_mode: str | None = typer.Option(None, help="W&B mode: online, offline, or disabled."),
    wandb_group: str | None = typer.Option(None, help="Optional W&B group."),
    wandb_run_name: str | None = typer.Option(None, help="Optional W&B run name."),
) -> None:
    """Run paper baseline strategies on synthetic data."""

    tracker = init_wandb_run(
        enabled=wandb,
        job_type="synthetic-baselines",
        config={
            "output_dir": output_dir,
            "days": days,
            "events_per_day": events_per_day,
            "episode_events": episode_events,
            "tabular_episodes": tabular_episodes,
            "seed": seed,
        },
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
        group=wandb_group,
        name=wandb_run_name,
        tags=("synthetic", "baselines"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, float | str | int]] = []
    trade_rows: list[dict[str, float | int | str]] = []

    for day_index, day in enumerate(PAPER_TRADING_DAYS_201911[:days]):
        dataset = generate_synthetic_lob_day(
            SyntheticLobConfig(day=day, n_events=events_per_day, seed=seed + day_index)
        )
        sigma = max(estimate_event_volatility(dataset), 1e-6)
        evaluation_events = min(episode_events, events_per_day - 1)
        strategies = [
            FixedLevelStrategy(level=1),
            FixedLevelStrategy(level=2),
            FixedLevelStrategy(level=3),
            RandomLevelStrategy(max_level=5, seed=seed + day_index),
            AvellanedaStoikovStrategy(sigma=sigma),
        ]
        for strategy in strategies:
            metrics, log_rows = evaluate_quote_strategy(
                dataset,
                strategy,
                episode_events=evaluation_events,
                latency_events=1,
                seed=seed + day_index,
            )
            metrics["day"] = day
            metrics["stock"] = dataset.stock
            metrics_rows.append(metrics)
            for row in log_rows:
                row["day"] = day
                row["stock"] = dataset.stock
            trade_rows.extend(log_rows)

        tabular_config = QLearningConfig(
            episodes=tabular_episodes,
            episode_events=evaluation_events,
            seed=seed + 10_000 + day_index,
        )
        for name, encoder, action_space in (
            ("Inv-RL", InventoryTimeEncoder(), OffsetActionSpace()),
            ("LOB-RL", LobRlEncoder(), BestBidAskActionSpace()),
        ):
            metrics, log_rows, strategy = train_and_evaluate_tabular_baseline(
                dataset,
                name=name,
                encoder=encoder,
                action_space=action_space,
                config=tabular_config,
            )
            metrics["day"] = day
            metrics["stock"] = dataset.stock
            metrics["q_states"] = len(strategy.q_table)
            metrics_rows.append(metrics)
            for row in log_rows:
                row["day"] = day
                row["stock"] = dataset.stock
            trade_rows.extend(log_rows)

    metrics_path = output_dir / "baseline_metrics.csv"
    trades_path = output_dir / "baseline_trades.parquet"
    pl.DataFrame(metrics_rows).write_csv(metrics_path)
    pl.DataFrame(trade_rows).write_parquet(trades_path)
    tracker.log_metrics(
        {"metrics_rows": len(metrics_rows), "trade_rows": len(trade_rows)},
        prefix="baselines",
    )
    tracker.log_artifact(
        [metrics_path, trades_path],
        name=f"synthetic-baselines-seed-{seed}",
        artifact_type="baselines",
    )
    tracker.finish()
    console.print(f"[green]wrote[/green] {metrics_path}")
    console.print(f"[green]wrote[/green] {trades_path}")


@app.command("run-synthetic-latency-baselines")
def run_synthetic_latency_baselines(
    output_dir: Path = typer.Option(Path("runs/synthetic-latency"), help="Output directory."),
    latencies: str = typer.Option("1,5,10,20,50,100", help="Comma-separated event latencies."),
    days: int = typer.Option(1, min=1, max=len(PAPER_TRADING_DAYS_201911), help="Synthetic days."),
    events_per_day: int = typer.Option(1_000, min=200, help="Events generated per synthetic day."),
    episode_events: int = typer.Option(500, min=100, help="Events evaluated per episode."),
    fixed_level: int = typer.Option(
        1, min=1, max=PAPER.lob_levels, help="Fixed level for Figure 2."
    ),
    include_tabular: bool = typer.Option(True, help="Include Inv-RL and LOB-RL latency rows."),
    tabular_episodes: int = typer.Option(
        10, min=1, help="Q-learning episodes for tabular baselines."
    ),
    paper_scale: bool = typer.Option(True, help="Use paper table scales in the latency figure."),
    seed: int = typer.Option(1, help="Base random seed."),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Log this run to W&B."),
    wandb_entity: str = typer.Option(DEFAULT_WANDB_ENTITY, help="W&B entity/team."),
    wandb_project: str = typer.Option(DEFAULT_WANDB_PROJECT, help="W&B project."),
    wandb_mode: str | None = typer.Option(None, help="W&B mode: online, offline, or disabled."),
    wandb_group: str | None = typer.Option(None, help="Optional W&B group."),
    wandb_run_name: str | None = typer.Option(None, help="Optional W&B run name."),
) -> None:
    """Run a Figure-2-style latency sweep for synthetic baselines."""

    tracker = init_wandb_run(
        enabled=wandb,
        job_type="synthetic-latency-baselines",
        config={
            "output_dir": output_dir,
            "latencies": latencies,
            "days": days,
            "events_per_day": events_per_day,
            "episode_events": episode_events,
            "fixed_level": fixed_level,
            "include_tabular": include_tabular,
            "tabular_episodes": tabular_episodes,
            "paper_scale": paper_scale,
            "seed": seed,
        },
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
        group=wandb_group,
        name=wandb_run_name,
        tags=("synthetic", "latency", "baselines"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    latency_values = _parse_int_list(latencies)
    metrics_rows: list[dict[str, float | str | int]] = []
    trade_rows: list[dict[str, float | int | str]] = []

    for day_index, day in enumerate(PAPER_TRADING_DAYS_201911[:days]):
        dataset = generate_synthetic_lob_day(
            SyntheticLobConfig(day=day, n_events=events_per_day, seed=seed + day_index)
        )
        sigma = max(estimate_event_volatility(dataset), 1e-6)
        evaluation_events = min(episode_events, events_per_day - 1)

        for latency in latency_values:
            strategies = [
                ("Fixed", FixedLevelStrategy(level=fixed_level)),
                ("Random", RandomLevelStrategy(max_level=5, seed=seed + latency + day_index)),
                ("AS", AvellanedaStoikovStrategy(sigma=sigma)),
            ]
            for method_name, strategy in strategies:
                metrics, log_rows = evaluate_quote_strategy(
                    dataset,
                    strategy,
                    episode_events=evaluation_events,
                    latency_events=latency,
                    seed=seed + day_index,
                )
                metrics["method"] = method_name
                metrics["latency_events"] = latency
                metrics["day"] = day
                metrics["stock"] = dataset.stock
                metrics_rows.append(metrics)
                for row in log_rows:
                    row["method"] = method_name
                    row["latency_events"] = latency
                    row["day"] = day
                    row["stock"] = dataset.stock
                trade_rows.extend(log_rows)

            if include_tabular:
                tabular_config = QLearningConfig(
                    episodes=tabular_episodes,
                    episode_events=evaluation_events,
                    seed=seed + 20_000 + latency + day_index,
                )
                for name, encoder, action_space in (
                    ("Inv-RL", InventoryTimeEncoder(), OffsetActionSpace()),
                    ("LOB-RL", LobRlEncoder(), BestBidAskActionSpace()),
                ):
                    metrics, log_rows, strategy = train_and_evaluate_tabular_baseline(
                        dataset,
                        name=name,
                        encoder=encoder,
                        action_space=action_space,
                        config=tabular_config,
                        latency_events=latency,
                    )
                    metrics["latency_events"] = latency
                    metrics["day"] = day
                    metrics["stock"] = dataset.stock
                    metrics["q_states"] = len(strategy.q_table)
                    metrics_rows.append(metrics)
                    for row in log_rows:
                        row["latency_events"] = latency
                        row["day"] = day
                        row["stock"] = dataset.stock
                    trade_rows.extend(log_rows)

    metrics = pl.DataFrame(metrics_rows)
    metrics_path = output_dir / "latency_metrics.csv"
    trades_path = output_dir / "latency_trades.parquet"
    figure_path = output_dir / "latency_figure.png"
    metrics.write_csv(metrics_path)
    pl.DataFrame(trade_rows).write_parquet(trades_path)
    plot_latency_figure(metrics, figure_path, paper_scale=paper_scale)
    tracker.log_metrics(
        {"metrics_rows": metrics.height, "trade_rows": len(trade_rows)},
        prefix="latency",
    )
    tracker.log_artifact(
        [metrics_path, trades_path, figure_path],
        name=f"synthetic-latency-baselines-seed-{seed}",
        artifact_type="latency",
    )
    tracker.finish()
    console.print(f"[green]wrote[/green] {metrics_path}")
    console.print(f"[green]wrote[/green] {trades_path}")
    console.print(f"[green]wrote[/green] {figure_path}")


@app.command("pretrain-synthetic-attn-lob")
def pretrain_synthetic_attn_lob(
    output_dir: Path = typer.Option(Path("runs/synthetic-pretrain"), help="Output directory."),
    events: int = typer.Option(1_000, min=200, help="Synthetic events."),
    epochs: int = typer.Option(1, min=1, help="Training epochs."),
    batch_size: int = typer.Option(64, min=1, help="Batch size."),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda."),
    seed: int = typer.Option(1, help="Random seed."),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Log this run to W&B."),
    wandb_entity: str = typer.Option(DEFAULT_WANDB_ENTITY, help="W&B entity/team."),
    wandb_project: str = typer.Option(DEFAULT_WANDB_PROJECT, help="W&B project."),
    wandb_mode: str | None = typer.Option(None, help="W&B mode: online, offline, or disabled."),
    wandb_group: str | None = typer.Option(None, help="Optional W&B group."),
    wandb_run_name: str | None = typer.Option(None, help="Optional W&B run name."),
) -> None:
    """Run a small Attn-LOB pretraining experiment on synthetic data."""

    with wandb_run(
        enabled=wandb,
        job_type="pretrain",
        config={
            "model_name": "Attn-LOB",
            "output_dir": output_dir,
            "events": events,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "seed": seed,
        },
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
        group=wandb_group,
        name=wandb_run_name,
        tags=("synthetic", "pretrain", "attn-lob"),
    ) as tracker:
        _run_synthetic_pretrain(
            model_name="Attn-LOB",
            output_dir=output_dir,
            events=events,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            seed=seed,
            tracker=tracker,
        )


@app.command("pretrain-synthetic")
def pretrain_synthetic(
    model_name: str = typer.Option("Attn-LOB", help="FC-LOB, Conv-LOB, DeepLOB, or Attn-LOB."),
    output_dir: Path = typer.Option(Path("runs/synthetic-pretrain"), help="Output directory."),
    events: int = typer.Option(1_000, min=200, help="Synthetic events."),
    epochs: int = typer.Option(1, min=1, help="Training epochs."),
    batch_size: int = typer.Option(64, min=1, help="Batch size."),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda."),
    seed: int = typer.Option(1, help="Random seed."),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Log this run to W&B."),
    wandb_entity: str = typer.Option(DEFAULT_WANDB_ENTITY, help="W&B entity/team."),
    wandb_project: str = typer.Option(DEFAULT_WANDB_PROJECT, help="W&B project."),
    wandb_mode: str | None = typer.Option(None, help="W&B mode: online, offline, or disabled."),
    wandb_group: str | None = typer.Option(None, help="Optional W&B group."),
    wandb_run_name: str | None = typer.Option(None, help="Optional W&B run name."),
) -> None:
    """Run one Table I pretraining model on synthetic data."""

    with wandb_run(
        enabled=wandb,
        job_type="pretrain",
        config={
            "model_name": model_name,
            "output_dir": output_dir,
            "events": events,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "seed": seed,
        },
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
        group=wandb_group,
        name=wandb_run_name,
        tags=("synthetic", "pretrain", model_name.lower()),
    ) as tracker:
        _run_synthetic_pretrain(
            model_name=model_name,
            output_dir=output_dir,
            events=events,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            seed=seed,
            tracker=tracker,
        )


@app.command("train-synthetic-ppo")
def train_synthetic_ppo(
    output_dir: Path = typer.Option(Path("runs/synthetic-ppo"), help="Output directory."),
    events: int = typer.Option(1_000, min=200, help="Synthetic events."),
    episode_events: int = typer.Option(500, min=100, help="Events per PPO episode."),
    latency_events: int = typer.Option(1, min=1, help="Event latency in replay."),
    total_timesteps: int = typer.Option(1_024, min=1, help="PPO training timesteps."),
    n_steps: int = typer.Option(128, min=2, help="PPO rollout length."),
    batch_size: int = typer.Option(64, min=1, help="PPO minibatch size."),
    n_epochs: int = typer.Option(4, min=1, help="PPO optimization epochs per rollout."),
    learning_rate: float = typer.Option(1e-4, min=1e-8, help="PPO learning rate."),
    gamma: float = typer.Option(0.99, min=0.0, max=1.0, help="PPO discount factor."),
    gae_lambda: float = typer.Option(
        0.95, min=0.0, max=1.0, help="PPO generalized advantage estimation lambda."
    ),
    clip_range: float = typer.Option(0.2, min=0.0, help="PPO clipping range."),
    ent_coef: float = typer.Option(0.0, min=0.0, help="PPO entropy coefficient."),
    vf_coef: float = typer.Option(0.5, min=0.0, help="PPO value-function loss coefficient."),
    max_grad_norm: float = typer.Option(0.5, min=0.0, help="PPO gradient clipping norm."),
    policy_log_std_init: float = typer.Option(
        0.0, help="Initial log standard deviation for the PPO Gaussian policy."
    ),
    encoder_checkpoint: Path | None = typer.Option(None, help="Optional Attn-LOB checkpoint."),
    freeze_encoder: bool = typer.Option(False, help="Freeze loaded Attn-LOB encoder weights."),
    lob_mode: str = typer.Option("attn", help="LOB feature mode: attn, mlp, or none."),
    use_dynamic_state: bool = typer.Option(True, help="Include the 24-dimensional dynamic state."),
    normalize_actions: bool = typer.Option(
        False, help="Expose [-1, 1] PPO actions and map them to paper [0, 1] actions."
    ),
    random_episode_starts: bool = typer.Option(
        False, help="Sample a new valid episode start on each training reset."
    ),
    eta: float = typer.Option(PAPER.eta_dampened_pnl, help="Paper DP reward coefficient."),
    zeta: float = typer.Option(PAPER.zeta_inventory_penalty, help="Paper inventory penalty weight."),
    device: str = typer.Option(
        "auto", help="Torch device for Stable-Baselines3, e.g. auto/cpu/cuda."
    ),
    seed: int = typer.Option(1, help="Random seed."),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Log this run to W&B."),
    wandb_entity: str = typer.Option(DEFAULT_WANDB_ENTITY, help="W&B entity/team."),
    wandb_project: str = typer.Option(DEFAULT_WANDB_PROJECT, help="W&B project."),
    wandb_mode: str | None = typer.Option(None, help="W&B mode: online, offline, or disabled."),
    wandb_group: str | None = typer.Option(None, help="Optional W&B group."),
    wandb_run_name: str | None = typer.Option(None, help="Optional W&B run name."),
) -> None:
    """Train a paper C-PPO smoke/experiment run on synthetic data."""

    if batch_size > n_steps:
        raise typer.BadParameter("batch-size must be <= n-steps for the single-env PPO runner")

    from stable_baselines3 import PPO

    tracker = init_wandb_run(
        enabled=wandb,
        job_type="train-c-ppo",
        config={
            "output_dir": output_dir,
            "events": events,
            "episode_events": episode_events,
            "latency_events": latency_events,
            "total_timesteps": total_timesteps,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "policy_log_std_init": policy_log_std_init,
            "encoder_checkpoint": encoder_checkpoint,
            "freeze_encoder": freeze_encoder,
            "lob_mode": lob_mode,
            "use_dynamic_state": use_dynamic_state,
            "normalize_actions": normalize_actions,
            "random_episode_starts": random_episode_starts,
            "eta": eta,
            "zeta": zeta,
            "device": device,
            "seed": seed,
        },
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
        group=wandb_group,
        name=wandb_run_name,
        tags=("synthetic", "rl", "c-ppo"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=events, seed=seed))
    env = PaperMarketMakingEnv(
        dataset,
        episode_events=min(episode_events, events - 1),
        latency_events=latency_events,
        normalize_actions=normalize_actions,
        random_episode_starts=random_episode_starts,
        eta=eta,
        zeta=zeta,
        seed=seed,
    )
    policy_kwargs = {
        "features_extractor_class": AttnLOBFeatureExtractor,
        "features_extractor_kwargs": {
            "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint else None,
            "freeze_encoder": freeze_encoder,
            "lob_mode": lob_mode,
            "use_dynamic_state": use_dynamic_state,
        },
        "log_std_init": policy_log_std_init,
    }
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=seed,
        device=device,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)

    model_path = output_dir / "c_ppo_model"
    model.save(model_path)

    metrics, trade_log = _evaluate_ppo_model(model, env, seed=seed + 1)
    metrics.update(
        {
            "method": "C-PPO",
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "policy_log_std_init": policy_log_std_init,
            "events": events,
            "episode_events": min(episode_events, events - 1),
            "latency_events": latency_events,
            "lob_mode": lob_mode,
            "use_dynamic_state": use_dynamic_state,
            "normalize_actions": normalize_actions,
            "random_episode_starts": random_episode_starts,
            "eta": eta,
            "zeta": zeta,
        }
    )
    metrics_path = output_dir / "c_ppo_metrics.csv"
    trades_path = output_dir / "c_ppo_trades.parquet"
    pl.DataFrame([metrics]).write_csv(metrics_path)
    pl.DataFrame(trade_log).write_parquet(trades_path)
    tracker.log_metrics(metrics, prefix="eval")
    tracker.update_summary(metrics, prefix="eval")
    tracker.log_artifact(
        [model_path.with_suffix(".zip"), metrics_path, trades_path],
        name=f"c-ppo-seed-{seed}",
        artifact_type="model",
    )
    tracker.finish()
    console.print(f"[green]wrote[/green] {model_path}.zip")
    console.print(f"[green]wrote[/green] {metrics_path}")
    console.print(f"[green]wrote[/green] {trades_path}")


@app.command("train-synthetic-ddqn")
def train_synthetic_ddqn(
    output_dir: Path = typer.Option(Path("runs/synthetic-ddqn"), help="Output directory."),
    events: int = typer.Option(1_000, min=200, help="Synthetic events."),
    episode_events: int = typer.Option(500, min=100, help="Events per D-DQN episode."),
    latency_events: int = typer.Option(1, min=1, help="Event latency in replay."),
    total_timesteps: int = typer.Option(1_000, min=1, help="D-DQN training timesteps."),
    learning_starts: int = typer.Option(100, min=0, help="Warmup steps before gradient updates."),
    buffer_size: int = typer.Option(10_000, min=1, help="Replay buffer size."),
    batch_size: int = typer.Option(32, min=1, help="Replay minibatch size."),
    target_update_interval: int = typer.Option(250, min=1, help="Target network sync interval."),
    learning_rate: float = typer.Option(1e-4, min=1e-8, help="D-DQN learning rate."),
    encoder_checkpoint: Path | None = typer.Option(None, help="Optional Attn-LOB checkpoint."),
    freeze_encoder: bool = typer.Option(False, help="Freeze loaded Attn-LOB encoder weights."),
    lob_mode: str = typer.Option("attn", help="LOB feature mode: attn, mlp, or none."),
    use_dynamic_state: bool = typer.Option(True, help="Include the 24-dimensional dynamic state."),
    random_episode_starts: bool = typer.Option(
        False, help="Sample a new valid episode start on each training reset."
    ),
    eta: float = typer.Option(PAPER.eta_dampened_pnl, help="Paper DP reward coefficient."),
    zeta: float = typer.Option(PAPER.zeta_inventory_penalty, help="Paper inventory penalty weight."),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda."),
    seed: int = typer.Option(1, help="Random seed."),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Log this run to W&B."),
    wandb_entity: str = typer.Option(DEFAULT_WANDB_ENTITY, help="W&B entity/team."),
    wandb_project: str = typer.Option(DEFAULT_WANDB_PROJECT, help="W&B project."),
    wandb_mode: str | None = typer.Option(None, help="W&B mode: online, offline, or disabled."),
    wandb_group: str | None = typer.Option(None, help="Optional W&B group."),
    wandb_run_name: str | None = typer.Option(None, help="Optional W&B run name."),
) -> None:
    """Train the paper's discrete dueling Double DQN agent on synthetic data."""

    tracker = init_wandb_run(
        enabled=wandb,
        job_type="train-d-dqn",
        config={
            "output_dir": output_dir,
            "events": events,
            "episode_events": episode_events,
            "latency_events": latency_events,
            "total_timesteps": total_timesteps,
            "learning_starts": learning_starts,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "target_update_interval": target_update_interval,
            "learning_rate": learning_rate,
            "encoder_checkpoint": encoder_checkpoint,
            "freeze_encoder": freeze_encoder,
            "lob_mode": lob_mode,
            "use_dynamic_state": use_dynamic_state,
            "random_episode_starts": random_episode_starts,
            "eta": eta,
            "zeta": zeta,
            "device": device,
            "seed": seed,
        },
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
        group=wandb_group,
        name=wandb_run_name,
        tags=("synthetic", "rl", "d-dqn"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=events, seed=seed))
    env = PaperDiscreteMarketMakingEnv(
        dataset,
        episode_events=min(episode_events, events - 1),
        latency_events=latency_events,
        random_episode_starts=random_episode_starts,
        eta=eta,
        zeta=zeta,
        seed=seed,
    )
    config = DuelingDQNConfig(
        total_timesteps=total_timesteps,
        learning_starts=learning_starts,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_interval=target_update_interval,
        learning_rate=learning_rate,
        seed=seed,
    )
    model, train_result = train_dueling_dqn(
        env,
        config=config,
        encoder_checkpoint=str(encoder_checkpoint) if encoder_checkpoint else None,
        freeze_encoder=freeze_encoder,
        lob_mode=lob_mode,
        use_dynamic_state=use_dynamic_state,
        device=device,
    )
    model_path = output_dir / "d_dqn_model.pt"
    save_dueling_dqn(model, model_path, config=config, train_result=train_result)

    metrics, trade_log = evaluate_dueling_dqn(model, env, seed=seed + 1, device=device)
    metrics.update(
        {
            "method": "D-DQN",
            "total_timesteps": total_timesteps,
            "updates": train_result.updates,
            "final_epsilon": train_result.final_epsilon,
            "events": events,
            "episode_events": min(episode_events, events - 1),
            "latency_events": latency_events,
            "lob_mode": lob_mode,
            "use_dynamic_state": use_dynamic_state,
            "random_episode_starts": random_episode_starts,
            "eta": eta,
            "zeta": zeta,
        }
    )
    metrics_path = output_dir / "d_dqn_metrics.csv"
    trades_path = output_dir / "d_dqn_trades.parquet"
    losses_path = output_dir / "d_dqn_losses.csv"
    pl.DataFrame([metrics]).write_csv(metrics_path)
    pl.DataFrame(trade_log).write_parquet(trades_path)
    pl.DataFrame({"loss": train_result.losses}).write_csv(losses_path)
    for step, loss in enumerate(train_result.losses):
        tracker.log({"train/loss": loss}, step=step)
    tracker.log_metrics(metrics, prefix="eval")
    tracker.update_summary(metrics, prefix="eval")
    tracker.log_metrics(
        {"updates": train_result.updates, "final_epsilon": train_result.final_epsilon},
        prefix="train",
    )
    tracker.log_artifact(
        [model_path, metrics_path, trades_path, losses_path],
        name=f"d-dqn-seed-{seed}",
        artifact_type="model",
    )
    tracker.finish()
    console.print(f"[green]wrote[/green] {model_path}")
    console.print(f"[green]wrote[/green] {metrics_path}")
    console.print(f"[green]wrote[/green] {trades_path}")
    console.print(f"[green]wrote[/green] {losses_path}")


@app.command("benchmark-runtime-synthetic")
def benchmark_runtime_synthetic(
    output_path: Path = typer.Option(Path("runs/runtime_metrics.csv"), help="Output CSV."),
    events: int = typer.Option(300, min=200, help="Synthetic events."),
    episode_events: int = typer.Option(200, min=100, help="Events per benchmark episode."),
    train_timesteps: int = typer.Option(32, min=1, help="Tiny train timesteps for train timing."),
    seed: int = typer.Option(1, help="Random seed."),
    device: str = typer.Option("cpu", help="Torch device for RL timing."),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Log this run to W&B."),
    wandb_entity: str = typer.Option(DEFAULT_WANDB_ENTITY, help="W&B entity/team."),
    wandb_project: str = typer.Option(DEFAULT_WANDB_PROJECT, help="W&B project."),
    wandb_mode: str | None = typer.Option(None, help="W&B mode: online, offline, or disabled."),
    wandb_group: str | None = typer.Option(None, help="Optional W&B group."),
    wandb_run_name: str | None = typer.Option(None, help="Optional W&B run name."),
) -> None:
    """Measure Table-III-style runtime on a synthetic smoke workload."""

    from stable_baselines3 import PPO

    tracker = init_wandb_run(
        enabled=wandb,
        job_type="runtime-benchmark",
        config={
            "output_path": output_path,
            "events": events,
            "episode_events": episode_events,
            "train_timesteps": train_timesteps,
            "seed": seed,
            "device": device,
        },
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
        group=wandb_group,
        name=wandb_run_name,
        tags=("synthetic", "runtime"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=events, seed=seed))
    evaluation_events = min(episode_events, events - 1)
    rows: list[dict[str, float | str | int]] = []

    sigma = max(estimate_event_volatility(dataset), 1e-6)
    for name, strategy in (
        ("Random", RandomLevelStrategy(max_level=5, seed=seed)),
        ("Fixed", FixedLevelStrategy(level=1)),
        ("AS", AvellanedaStoikovStrategy(sigma=sigma)),
    ):
        start = perf_counter()
        _, log_rows = evaluate_quote_strategy(
            dataset,
            strategy,
            episode_events=evaluation_events,
            latency_events=1,
            seed=seed,
        )
        elapsed_ms = (perf_counter() - start) * 1000.0
        rows.append(
            {
                "method": name,
                "phase": "infer",
                "runtime_ms_per_ts": elapsed_ms / max(1, len(log_rows)),
                "timesteps": len(log_rows),
            }
        )

    ppo_env = PaperMarketMakingEnv(dataset, episode_events=evaluation_events, seed=seed)
    ppo_model = PPO(
        "MultiInputPolicy",
        ppo_env,
        policy_kwargs={"features_extractor_class": AttnLOBFeatureExtractor},
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        device=device,
        seed=seed,
        verbose=0,
    )
    start = perf_counter()
    ppo_model.learn(total_timesteps=train_timesteps)
    train_elapsed_ms = (perf_counter() - start) * 1000.0
    rows.append(
        {
            "method": "C-PPO",
            "phase": "train",
            "runtime_ms_per_ts": train_elapsed_ms / train_timesteps,
            "timesteps": train_timesteps,
        }
    )
    start = perf_counter()
    _, ppo_log = _evaluate_ppo_model(ppo_model, ppo_env, seed=seed + 1)
    infer_elapsed_ms = (perf_counter() - start) * 1000.0
    rows.append(
        {
            "method": "C-PPO",
            "phase": "infer",
            "runtime_ms_per_ts": infer_elapsed_ms / max(1, len(ppo_log)),
            "timesteps": len(ppo_log),
        }
    )

    ddqn_env = PaperDiscreteMarketMakingEnv(dataset, episode_events=evaluation_events, seed=seed)
    ddqn_config = DuelingDQNConfig(
        total_timesteps=train_timesteps,
        learning_starts=min(8, train_timesteps // 2),
        buffer_size=128,
        batch_size=8,
        target_update_interval=max(8, train_timesteps // 2),
        seed=seed,
    )
    start = perf_counter()
    ddqn_model, _ = train_dueling_dqn(ddqn_env, config=ddqn_config, device=device)
    train_elapsed_ms = (perf_counter() - start) * 1000.0
    rows.append(
        {
            "method": "D-DQN",
            "phase": "train",
            "runtime_ms_per_ts": train_elapsed_ms / train_timesteps,
            "timesteps": train_timesteps,
        }
    )
    start = perf_counter()
    _, ddqn_log = evaluate_dueling_dqn(ddqn_model, ddqn_env, seed=seed + 1, device=device)
    infer_elapsed_ms = (perf_counter() - start) * 1000.0
    rows.append(
        {
            "method": "D-DQN",
            "phase": "infer",
            "runtime_ms_per_ts": infer_elapsed_ms / max(1, len(ddqn_log)),
            "timesteps": len(ddqn_log),
        }
    )

    pl.DataFrame(rows).write_csv(output_path)
    for row in rows:
        method = str(row["method"]).lower().replace("-", "_")
        phase = str(row["phase"])
        tracker.log_metrics(row, prefix=f"runtime/{method}/{phase}")
    tracker.log_artifact(output_path, name=f"runtime-seed-{seed}", artifact_type="runtime")
    tracker.finish()
    console.print(f"[green]wrote[/green] {output_path}")


@app.command("run-full-synthetic-replication")
def run_full_synthetic_replication(
    output_dir: Path = typer.Option(
        Path("runs/full-synthetic-replication"), help="Output directory."
    ),
    stocks: str = typer.Option(
        "000001,000858,002415", help="Comma-separated synthetic stock codes."
    ),
    base_prices: str | None = typer.Option(
        None, help="Comma-separated base prices matching --stocks."
    ),
    train_days: int = typer.Option(
        PAPER_TRAIN_DAYS,
        min=1,
        max=len(PAPER_TRADING_DAYS_201911) - 1,
        help="Synthetic train days from the first half of November 2019.",
    ),
    test_days: int = typer.Option(
        PAPER_TEST_DAYS,
        min=1,
        max=len(PAPER_TRADING_DAYS_201911) - 1,
        help="Synthetic test days after the train split.",
    ),
    events_per_day: int = typer.Option(
        FULL_REPLICATION_EVENTS_PER_DAY,
        min=300,
        help="Events generated per synthetic stock/day before stable-window filtering.",
    ),
    episode_events: int = typer.Option(
        PAPER.episode_events, min=100, help="Events evaluated per episode."
    ),
    pretrain_events: int = typer.Option(
        0, min=0, help="Maximum first-stock train events for Table I; 0 uses all train events."
    ),
    pretrain_epochs: int = typer.Option(1, min=1, help="Pretraining epochs per Table I model."),
    pretrain_batch_size: int = typer.Option(64, min=1, help="Pretraining batch size."),
    agent_timesteps: int = typer.Option(
        FULL_REPLICATION_AGENT_TIMESTEPS,
        min=1,
        help="C-PPO and D-DQN train timesteps.",
    ),
    tabular_episodes: int = typer.Option(
        PAPER_TRAIN_DAYS * 5, min=1, help="Inv-RL/LOB-RL Q-learning episodes."
    ),
    latency_values: str = typer.Option("1,5,10,20,50,100", help="Latency grid."),
    runtime_train_timesteps: int = typer.Option(32, min=1, help="Tiny train timing steps."),
    seed: int = typer.Option(1, help="Base random seed."),
    device: str = typer.Option("cpu", help="Torch/SB3 device."),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Log this run to W&B."),
    wandb_entity: str = typer.Option(DEFAULT_WANDB_ENTITY, help="W&B entity/team."),
    wandb_project: str = typer.Option(DEFAULT_WANDB_PROJECT, help="W&B project."),
    wandb_mode: str | None = typer.Option(None, help="W&B mode: online, offline, or disabled."),
    wandb_group: str | None = typer.Option(None, help="Optional W&B group."),
    wandb_run_name: str | None = typer.Option(None, help="Optional W&B run name."),
) -> None:
    """Run the authoritative synthetic version of all paper tables and figures.

    The paper's proprietary exchange data is not an input to this project. This
    command therefore fixes the data substitute to synthetic canonical replay
    data and emits reproducible artifacts for the paper's Table I-IV and
    Figure 2-4 surfaces.
    """

    from stable_baselines3 import PPO

    output_dir.mkdir(parents=True, exist_ok=True)
    if train_days + test_days > len(PAPER_TRADING_DAYS_201911):
        raise typer.BadParameter("--train-days + --test-days cannot exceed 21 trading days")
    stock_specs = _parse_synthetic_stock_specs(stocks, base_prices)
    latencies = _parse_int_list(latency_values)
    tracker = init_wandb_run(
        enabled=wandb,
        job_type="full-synthetic-replication",
        config={
            "output_dir": output_dir,
            "stocks": [stock for stock, _ in stock_specs],
            "base_prices": [price for _, price in stock_specs],
            "train_days": train_days,
            "test_days": test_days,
            "events_per_day": events_per_day,
            "episode_events": episode_events,
            "pretrain_events": pretrain_events,
            "pretrain_epochs": pretrain_epochs,
            "pretrain_batch_size": pretrain_batch_size,
            "agent_timesteps": agent_timesteps,
            "tabular_episodes": tabular_episodes,
            "latencies": latencies,
            "runtime_train_timesteps": runtime_train_timesteps,
            "seed": seed,
            "device": device,
        },
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
        group=wandb_group,
        name=wandb_run_name,
        tags=("synthetic", "replication", "paper"),
    )
    train_day_indices = list(range(train_days))
    test_day_indices = list(range(train_days, train_days + test_days))
    train_datasets = _build_synthetic_panel(
        output_dir=output_dir / "data" / "train",
        stock_specs=stock_specs,
        day_indices=train_day_indices,
        events_per_day=events_per_day,
        seed=seed,
        stable_windows_only=True,
    )
    test_datasets = _build_synthetic_panel(
        output_dir=output_dir / "data" / "test",
        stock_specs=stock_specs,
        day_indices=test_day_indices,
        events_per_day=events_per_day,
        seed=seed,
        stable_windows_only=True,
    )
    first_test_dataset = test_datasets[0][2]

    config_path = output_dir / "replication_config.md"
    _write_full_replication_config(
        config_path,
        stock_specs=stock_specs,
        train_days=train_days,
        test_days=test_days,
        events_per_day=events_per_day,
        episode_events=episode_events,
        pretrain_events=pretrain_events,
        pretrain_epochs=pretrain_epochs,
        agent_timesteps=agent_timesteps,
        tabular_episodes=tabular_episodes,
        latencies=latencies,
        seed=seed,
    )

    table_i_rows = []
    pretrain_model_paths: list[Path] = []
    table_i_dir = output_dir / "table_i_pretraining"
    pretrain_dataset = _stock_dataset_from_panel(
        train_datasets,
        stock=stock_specs[0][0],
        max_events=pretrain_events,
        day="train",
    )
    pretrain_evaluation_dataset = _stock_dataset_from_panel(
        test_datasets,
        stock=stock_specs[0][0],
        max_events=pretrain_events,
        day="test",
    )
    attn_lob_checkpoint: Path | None = None
    for model_index, model_name in enumerate(PRETRAIN_MODEL_NAMES):
        row, model_path = _run_pretrain_on_dataset(
            dataset=pretrain_dataset,
            evaluation_dataset=pretrain_evaluation_dataset,
            model_name=model_name,
            output_dir=table_i_dir,
            epochs=pretrain_epochs,
            batch_size=pretrain_batch_size,
            device=device,
            seed=seed + model_index,
        )
        if model_name == "Attn-LOB":
            attn_lob_checkpoint = model_path
        pretrain_model_paths.append(model_path)
        table_i_rows.append(row)
    if attn_lob_checkpoint is None:
        raise RuntimeError("Attn-LOB pretraining did not produce a checkpoint")
    table_i_path = table_i_dir / "table_i_pretrain_metrics.csv"
    pl.DataFrame(table_i_rows).write_csv(table_i_path)

    overall_metrics, overall_trades, first_ppo_trade_log, first_ppo_model = (
        _run_overall_synthetic_table(
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            output_dir=output_dir / "table_ii_overall",
            episode_events=episode_events,
            tabular_episodes=tabular_episodes,
            agent_timesteps=agent_timesteps,
            encoder_checkpoint=attn_lob_checkpoint,
            seed=seed,
            device=device,
            ppo_class=PPO,
        )
    )
    overall_metrics_path = output_dir / "table_ii_overall" / "overall_metrics.csv"
    overall_trades_path = output_dir / "table_ii_overall" / "overall_trades.parquet"
    overall_summary_path = output_dir / "table_ii_overall" / "overall_summary.csv"
    pl.DataFrame(overall_metrics).write_csv(overall_metrics_path)
    pl.DataFrame(overall_trades).write_parquet(overall_trades_path)
    summarize_paper_table(pl.DataFrame(overall_metrics)).write_csv(overall_summary_path)

    latency_metrics, latency_trades = _run_latency_synthetic_table(
        train_dataset=_first_stock_train_dataset(
            train_datasets,
            stock=stock_specs[0][0],
            max_events=0,
        ),
        test_dataset=first_test_dataset,
        output_dir=output_dir / "figure_2_latency",
        latencies=latencies,
        episode_events=episode_events,
        tabular_episodes=tabular_episodes,
        agent_timesteps=agent_timesteps,
        encoder_checkpoint=attn_lob_checkpoint,
        seed=seed,
        device=device,
        ppo_class=PPO,
    )
    latency_dir = output_dir / "figure_2_latency"
    latency_metrics_path = latency_dir / "latency_metrics.csv"
    latency_trades_path = latency_dir / "latency_trades.parquet"
    latency_figure_path = latency_dir / "figure_2_latency.png"
    pl.DataFrame(latency_metrics).write_csv(latency_metrics_path)
    pl.DataFrame(latency_trades).write_parquet(latency_trades_path)
    plot_latency_figure(pl.DataFrame(latency_metrics), latency_figure_path)

    runtime_path = output_dir / "table_iii_runtime" / "runtime_metrics.csv"
    benchmark_runtime_synthetic(
        output_path=runtime_path,
        events=min(events_per_day, 1_000),
        episode_events=min(episode_events, events_per_day - 1),
        train_timesteps=runtime_train_timesteps,
        seed=seed,
        device=device,
    )

    ablation_metrics, ablation_trades = _run_ablation_synthetic_table(
        train_dataset=_first_stock_train_dataset(
            train_datasets,
            stock=stock_specs[0][0],
            max_events=0,
        ),
        test_dataset=first_test_dataset,
        output_dir=output_dir / "table_iv_ablation",
        episode_events=episode_events,
        agent_timesteps=agent_timesteps,
        encoder_checkpoint=attn_lob_checkpoint,
        seed=seed,
        device=device,
        ppo_class=PPO,
    )
    ablation_dir = output_dir / "table_iv_ablation"
    ablation_metrics_path = ablation_dir / "ablation_metrics.csv"
    ablation_summary_path = ablation_dir / "ablation_summary.csv"
    ablation_trades_path = ablation_dir / "ablation_trades.parquet"
    pl.DataFrame(ablation_metrics).write_csv(ablation_metrics_path)
    pl.DataFrame(ablation_trades).write_parquet(ablation_trades_path)
    summarize_paper_table(
        pl.DataFrame(ablation_metrics),
        group_columns=("method", "variant", "stock"),
    ).write_csv(ablation_summary_path)

    if not first_ppo_trade_log:
        raise RuntimeError("no held-out C-PPO episode was available for Figure 4")
    figure_3_path = output_dir / "figure_3_attention" / "figure_3_attention.png"
    _plot_attention_from_ppo_encoder(first_ppo_model, first_test_dataset, figure_3_path)
    figure_4_path = output_dir / "figure_4_decision_trace" / "figure_4_decision_trace.png"
    plot_decision_trace(pl.DataFrame(first_ppo_trade_log), figure_4_path)

    _write_full_replication_index(
        output_dir / "README.md",
        paths=[
            config_path,
            table_i_path,
            overall_metrics_path,
            overall_summary_path,
            latency_metrics_path,
            latency_figure_path,
            runtime_path,
            ablation_metrics_path,
            ablation_summary_path,
            figure_3_path,
            figure_4_path,
        ],
    )
    for row in table_i_rows:
        model_prefix = str(row["model"]).lower().replace("-", "_")
        tracker.log_metrics(row, prefix=f"table_i/{model_prefix}")
        tracker.update_summary(row, prefix=f"table_i/{model_prefix}")
    tracker.log_metrics(
        {
            "table_i_rows": len(table_i_rows),
            "overall_rows": len(overall_metrics),
            "overall_trade_rows": len(overall_trades),
            "latency_rows": len(latency_metrics),
            "latency_trade_rows": len(latency_trades),
            "ablation_rows": len(ablation_metrics),
            "ablation_trade_rows": len(ablation_trades),
        },
        prefix="replication",
    )
    tracker.log_artifact(
        [
            output_dir / "README.md",
            config_path,
            table_i_path,
            overall_metrics_path,
            overall_summary_path,
            latency_metrics_path,
            latency_figure_path,
            runtime_path,
            ablation_metrics_path,
            ablation_summary_path,
            figure_3_path,
            figure_4_path,
            *pretrain_model_paths,
        ],
        name=f"full-synthetic-replication-seed-{seed}",
        artifact_type="replication",
    )
    tracker.finish()
    console.print(f"[green]wrote full synthetic replication[/green] {output_dir}")


@app.command("collect-metrics")
def collect_metrics(
    input_glob: str = typer.Argument(..., help="Glob for metrics CSV files."),
    output_path: Path = typer.Option(Path("runs/collected_metrics.csv"), help="Output CSV."),
) -> None:
    """Collect multiple metrics CSV files into one dataframe."""

    import glob

    paths = [Path(path) for path in sorted(glob.glob(input_glob, recursive=True))]
    if not paths:
        raise typer.BadParameter(f"no files matched: {input_glob}")
    frames = [_read_metrics_csv(path) for path in paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames, how="diagonal_relaxed").write_csv(output_path)
    console.print(f"[green]wrote[/green] {output_path} ({len(paths)} files)")


@app.command("summarize-metrics")
def summarize_metrics(
    metrics_path: Path = typer.Argument(..., help="Input metrics CSV."),
    output_path: Path = typer.Option(Path("runs/summary_metrics.csv"), help="Output summary CSV."),
) -> None:
    """Aggregate repeated runs into paper-scaled table columns."""

    metrics = _read_metrics_csv(metrics_path)
    summary = summarize_paper_table(metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(output_path)
    console.print(f"[green]wrote[/green] {output_path}")


@app.command("plot-latency-figure")
def plot_latency_command(
    metrics_path: Path = typer.Argument(..., help="Latency metrics CSV."),
    output_path: Path = typer.Option(Path("runs/latency_figure.png"), help="Output PNG."),
    paper_scale: bool = typer.Option(True, help="Use paper table scales in the plot."),
) -> None:
    """Plot a Figure-2-style latency figure from metrics."""

    plot_latency_figure(_read_metrics_csv(metrics_path), output_path, paper_scale=paper_scale)
    console.print(f"[green]wrote[/green] {output_path}")


@app.command("plot-decision-trace")
def plot_decision_command(
    trades_path: Path = typer.Argument(..., help="Trade log Parquet or CSV."),
    output_path: Path = typer.Option(Path("runs/decision_trace.png"), help="Output PNG."),
) -> None:
    """Plot a Figure-4-style decision trace from a trade log."""

    plot_decision_trace(_read_table(trades_path), output_path)
    console.print(f"[green]wrote[/green] {output_path}")


@app.command("plot-synthetic-attention")
def plot_synthetic_attention(
    output_path: Path = typer.Option(Path("runs/attention_heatmap.png"), help="Output PNG."),
    checkpoint: Path | None = typer.Option(None, help="Optional Attn-LOB classifier checkpoint."),
    events: int = typer.Option(300, min=100, help="Synthetic events."),
    index: int = typer.Option(80, min=PAPER.window_length - 1, help="Event index to visualize."),
    seed: int = typer.Option(1, help="Random seed."),
) -> None:
    """Plot a Figure-3-style attention heatmap for one synthetic LOB window."""

    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=events, seed=seed))
    if index >= dataset.orderbook.height:
        raise typer.BadParameter("index must be smaller than the generated event count")
    model = AttnLOBClassifier()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)
    model.eval()

    start = index - PAPER.window_length + 1
    lob_values = dataset.orderbook.select(lob_columns()).slice(start, PAPER.window_length)
    window = normalize_lob_window(lob_values.to_numpy())
    with torch.no_grad():
        _, weights = model.encoder(
            torch.from_numpy(window).float().unsqueeze(0),
            return_attention_weights=True,
        )
    plot_attention_heatmap(weights.squeeze(0).numpy(), output_path, lob_window=window)
    console.print(f"[green]wrote[/green] {output_path}")


def _parse_synthetic_stock_specs(
    stocks: str,
    base_prices: str | None,
    *,
    fallback: float | None = None,
) -> list[tuple[str, float]]:
    stock_codes = [part.strip() for part in stocks.split(",") if part.strip()]
    if not stock_codes:
        raise typer.BadParameter("expected at least one stock code")
    if base_prices:
        prices = [float(part.strip()) for part in base_prices.split(",") if part.strip()]
        if len(prices) != len(stock_codes):
            raise typer.BadParameter("--base-prices must match the number of stocks")
    else:
        prices = [
            SYNTHETIC_STOCK_BASE_PRICES.get(
                stock_code,
                fallback if fallback is not None else 16.45,
            )
            for stock_code in stock_codes
        ]
    return list(zip(stock_codes, prices, strict=True))


def _generate_configured_synthetic_day(
    *,
    stock: str,
    day: str,
    events: int,
    base_price: float,
    seed: int,
):
    return generate_synthetic_lob_day(
        SyntheticLobConfig(
            stock=stock,
            day=day,
            n_events=events,
            base_price=base_price,
            seed=seed,
        )
    )


def _build_synthetic_panel(
    *,
    output_dir: Path,
    stock_specs: list[tuple[str, float]],
    day_indices: list[int],
    events_per_day: int,
    seed: int,
    stable_windows_only: bool = False,
) -> list[tuple[str, str, LobDataset]]:
    panel: list[tuple[str, str, LobDataset]] = []
    for stock_index, (stock, base_price) in enumerate(stock_specs):
        for day_index in day_indices:
            day = PAPER_TRADING_DAYS_201911[day_index]
            dataset = _generate_configured_synthetic_day(
                stock=stock,
                day=day,
                events=events_per_day,
                base_price=base_price,
                seed=seed + 1_000 * stock_index + day_index,
            )
            if stable_windows_only:
                dataset = _filter_stable_windows(dataset)
            write_lob_dataset(dataset, output_dir)
            panel.append((stock, day, dataset))
    return panel


def _filter_stable_windows(dataset: LobDataset) -> LobDataset:
    starts_and_ends = [
        (time.fromisoformat(start), time.fromisoformat(end))
        for start, end in PAPER_PRETRAIN_WINDOWS
    ]
    mask = [
        any(start <= timestamp.time() <= end for start, end in starts_and_ends)
        for timestamp in dataset.orderbook["timestamp"].to_list()
    ]
    return LobDataset(
        stock=dataset.stock,
        day=dataset.day,
        orderbook=dataset.orderbook.filter(mask),
        messages=dataset.messages.filter(mask),
        trades=dataset.trades.filter(mask),
    )


def _first_stock_train_dataset(
    panel: list[tuple[str, str, LobDataset]],
    *,
    stock: str,
    max_events: int,
) -> LobDataset:
    return _stock_dataset_from_panel(panel, stock=stock, max_events=max_events, day="train")


def _stock_dataset_from_panel(
    panel: list[tuple[str, str, LobDataset]],
    *,
    stock: str,
    max_events: int,
    day: str,
) -> LobDataset:
    datasets = [dataset for panel_stock, _, dataset in panel if panel_stock == stock]
    if not datasets:
        raise ValueError(f"no datasets found for stock {stock}")
    merged = _merge_lob_datasets(datasets, day=day)
    if max_events > 0 and merged.orderbook.height > max_events:
        merged = LobDataset(
            stock=merged.stock,
            day=merged.day,
            orderbook=merged.orderbook.head(max_events),
            messages=merged.messages.head(max_events),
            trades=merged.trades.head(max_events),
        )
    return merged


def _merge_lob_datasets(datasets: list[LobDataset], *, day: str) -> LobDataset:
    if not datasets:
        raise ValueError("expected at least one dataset to merge")
    stock = datasets[0].stock
    if any(dataset.stock != stock for dataset in datasets):
        raise ValueError("cannot merge datasets from different stocks")
    return LobDataset(
        stock=stock,
        day=day,
        orderbook=pl.concat([dataset.orderbook for dataset in datasets], how="vertical"),
        messages=pl.concat([dataset.messages for dataset in datasets], how="vertical"),
        trades=pl.concat([dataset.trades for dataset in datasets], how="vertical"),
    )


def _panel_by_stock(
    panel: list[tuple[str, str, LobDataset]],
) -> dict[str, list[tuple[str, LobDataset]]]:
    grouped: dict[str, list[tuple[str, LobDataset]]] = {}
    for stock, day, dataset in panel:
        grouped.setdefault(stock, []).append((day, dataset))
    return grouped


def _episode_starts(dataset: LobDataset, *, episode_events: int, latency_events: int) -> list[int]:
    if episode_events <= PAPER.window_length + latency_events:
        return []

    starts_and_ends = [
        (time.fromisoformat(start), time.fromisoformat(end))
        for start, end in PAPER_PRETRAIN_WINDOWS
    ]
    timestamps = dataset.orderbook["timestamp"].to_list()
    starts: list[int] = []
    for window_start, window_end in starts_and_ends:
        indices = [
            index
            for index, timestamp in enumerate(timestamps)
            if window_start <= timestamp.time() <= window_end
        ]
        if not indices:
            continue
        window_first = indices[0]
        window_last_exclusive = indices[-1] + 1
        start = window_first
        while start + episode_events <= window_last_exclusive:
            starts.append(start)
            start += episode_events

    if starts:
        return starts
    if dataset.orderbook.height > episode_events:
        return [0]
    return []


def _run_pretrain_on_dataset(
    *,
    dataset: LobDataset,
    evaluation_dataset: LobDataset | None = None,
    model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    input_shape = pretrain_input_shape(model_name)
    arrays = build_pretrain_arrays(dataset, window_length=input_shape[0])
    evaluation_arrays = (
        build_pretrain_arrays(evaluation_dataset, window_length=input_shape[0])
        if evaluation_dataset is not None
        else None
    )
    model = make_pretrain_model(model_name)
    metrics = train_lob_classifier(
        model,
        arrays,
        evaluation_arrays=evaluation_arrays,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    safe_model_name = model_name.lower().replace("-", "_")
    implementation_param = count_parameters(model)
    implementation_encoder_param = count_encoder_parameters(model)
    paper_param = paper_reported_parameter_count(model_name)
    row = {
        "model": model_name,
        "stock": dataset.stock,
        "day": dataset.day,
        "evaluation_day": evaluation_dataset.day if evaluation_dataset else dataset.day,
        **metrics.__dict__,
        "input_window_length": input_shape[0],
        "implementation_param": implementation_param,
        "implementation_encoder_param": implementation_encoder_param,
        "paper_reported_param": paper_param,
        "param_matches_paper_report": implementation_encoder_param == paper_param,
        "full_param_matches_paper_report": implementation_param == paper_param,
        "encoder_param_matches_paper_report": implementation_encoder_param == paper_param,
    }
    metrics_path = output_dir / f"{safe_model_name}_pretrain_metrics.csv"
    model_path = output_dir / f"{safe_model_name}_pretrain_model.pt"
    pl.DataFrame([row]).write_csv(metrics_path)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"model": model_name, "state_dict": state_dict}, model_path)
    return row, model_path


def _run_overall_synthetic_table(
    *,
    train_datasets: list[tuple[str, str, LobDataset]],
    test_datasets: list[tuple[str, str, LobDataset]],
    output_dir: Path,
    episode_events: int,
    tabular_episodes: int,
    agent_timesteps: int,
    encoder_checkpoint: Path,
    seed: int,
    device: str,
    ppo_class,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, float | int | str | bool]] = []
    trade_rows: list[dict[str, float | int | str]] = []
    first_ppo_trade_log: list[dict[str, float | int]] = []
    first_ppo_model: object | None = None
    train_by_stock = _panel_by_stock(train_datasets)
    test_by_stock = _panel_by_stock(test_datasets)

    for stock_index, (stock, stock_train_entries) in enumerate(train_by_stock.items()):
        stock_test_entries = test_by_stock.get(stock)
        if not stock_test_entries:
            raise ValueError(f"no held-out test datasets found for stock {stock}")

        train_dataset = _merge_lob_datasets(
            [dataset for _, dataset in stock_train_entries],
            day="train",
        )
        baseline_strategies = _baseline_strategies_for_train_data(
            train_dataset,
            episode_events=episode_events,
            tabular_episodes=tabular_episodes,
            seed=seed + 10_000 * stock_index,
        )
        ppo_model = _train_ppo_model(
            train_dataset,
            output_dir=output_dir / stock / "c_ppo",
            episode_events=episode_events,
            latency_events=1,
            total_timesteps=agent_timesteps,
            seed=seed + 30_000 + stock_index,
            device=device,
            ppo_class=ppo_class,
            encoder_checkpoint=encoder_checkpoint,
            normalize_actions=True,
            policy_log_std_init=FULL_REPLICATION_PPO_LOG_STD_INIT,
        )
        if first_ppo_model is None:
            first_ppo_model = ppo_model

        ddqn_model = _train_ddqn_model(
            train_dataset,
            output_dir=output_dir / stock / "d_dqn",
            episode_events=episode_events,
            latency_events=1,
            total_timesteps=agent_timesteps,
            seed=seed + 40_000 + stock_index,
            device=device,
            encoder_checkpoint=encoder_checkpoint,
        )

        for test_index, (day, test_dataset) in enumerate(stock_test_entries):
            episode_seed = seed + 1_000 * stock_index + test_index
            for row, log_rows in _evaluate_strategies_on_test_episodes(
                test_dataset,
                strategies=baseline_strategies,
                episode_events=episode_events,
                latency_events=1,
                seed=episode_seed,
            ):
                row.update({"stock": stock, "day": day})
                metrics_rows.append(row)
                _extend_tagged_trades(
                    trade_rows,
                    log_rows,
                    stock=stock,
                    day=day,
                    method=str(row["method"]),
                )

            for row, log_rows in _evaluate_ppo_on_test_episodes(
                ppo_model,
                test_dataset,
                episode_events=episode_events,
                latency_events=1,
                normalize_actions=True,
                seed=episode_seed + 30_000,
            ):
                row.update(
                    {
                        "method": "C-PPO",
                        "stock": stock,
                        "day": day,
                        "total_timesteps": agent_timesteps,
                        "latency_events": 1,
                        "lob_mode": "attn",
                        "use_dynamic_state": True,
                        "normalize_actions": True,
                        "policy_log_std_init": FULL_REPLICATION_PPO_LOG_STD_INIT,
                        "encoder_checkpoint": str(encoder_checkpoint),
                    }
                )
                metrics_rows.append(row)
                _extend_tagged_trades(
                    trade_rows,
                    log_rows,
                    stock=stock,
                    day=day,
                    method="C-PPO",
                )
                if not first_ppo_trade_log:
                    first_ppo_trade_log = log_rows

            for row, log_rows in _evaluate_ddqn_on_test_episodes(
                ddqn_model,
                test_dataset,
                episode_events=episode_events,
                latency_events=1,
                seed=episode_seed + 40_000,
                device=device,
            ):
                row.update(
                    {
                        "method": "D-DQN",
                        "stock": stock,
                        "day": day,
                        "total_timesteps": agent_timesteps,
                        "latency_events": 1,
                        "lob_mode": "attn",
                        "use_dynamic_state": True,
                        "encoder_checkpoint": str(encoder_checkpoint),
                    }
                )
                metrics_rows.append(row)
                _extend_tagged_trades(
                    trade_rows,
                    log_rows,
                    stock=stock,
                    day=day,
                    method="D-DQN",
                )

    if first_ppo_model is None:
        raise RuntimeError("overall replication did not train a PPO model")
    return metrics_rows, trade_rows, first_ppo_trade_log, first_ppo_model


def _baseline_strategies_for_train_data(
    train_dataset: LobDataset,
    *,
    episode_events: int,
    tabular_episodes: int,
    seed: int,
) -> list[object]:
    sigma = max(estimate_event_volatility(train_dataset), 1e-6)
    strategies: list[object] = [
        FixedLevelStrategy(level=1),
        FixedLevelStrategy(level=2),
        FixedLevelStrategy(level=3),
        RandomLevelStrategy(max_level=5, seed=seed),
        AvellanedaStoikovStrategy(sigma=sigma),
    ]
    tabular_config = QLearningConfig(
        episodes=tabular_episodes,
        episode_events=episode_events,
        seed=seed + 10_000,
    )
    for name, encoder, action_space in (
        ("Inv-RL", InventoryTimeEncoder(), OffsetActionSpace()),
        ("LOB-RL", LobRlEncoder(), BestBidAskActionSpace()),
    ):
        strategy = train_tabular_q_strategy(
            train_dataset,
            name=name,
            encoder=encoder,
            action_space=action_space,
            config=tabular_config,
        )
        strategies.append(strategy)
    return strategies


def _evaluate_strategies_on_test_episodes(
    dataset: LobDataset,
    *,
    strategies: list[object],
    episode_events: int,
    latency_events: int,
    seed: int,
) -> list[tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]]:
    rows: list[tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]] = []
    evaluation_events = min(episode_events, dataset.orderbook.height - 1)
    starts = _episode_starts(
        dataset,
        episode_events=evaluation_events,
        latency_events=latency_events,
    )
    for strategy in strategies:
        for episode_id, episode_start in enumerate(starts):
            metrics, log_rows = evaluate_quote_strategy(
                dataset,
                strategy,  # type: ignore[arg-type]
                episode_start=episode_start,
                episode_events=evaluation_events,
                latency_events=latency_events,
                seed=seed + episode_id,
            )
            metrics.update(
                {
                    "episode_id": episode_id,
                    "episode_start": episode_start,
                    "episode_events": evaluation_events,
                    "latency_events": latency_events,
                }
            )
            if hasattr(strategy, "q_table"):
                metrics["q_states"] = len(strategy.q_table)  # type: ignore[attr-defined]
            for row in log_rows:
                row["episode_id"] = episode_id
                row["episode_start"] = episode_start
            rows.append((metrics, log_rows))
    return rows


def _train_ppo_model(
    train_dataset: LobDataset,
    *,
    output_dir: Path,
    episode_events: int,
    latency_events: int,
    total_timesteps: int,
    seed: int,
    device: str,
    ppo_class,
    lob_mode: str = "attn",
    use_dynamic_state: bool = True,
    encoder_checkpoint: Path | None = None,
    freeze_encoder: bool = False,
    normalize_actions: bool = True,
    policy_log_std_init: float = FULL_REPLICATION_PPO_LOG_STD_INIT,
) -> object:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = PaperMarketMakingEnv(
        train_dataset,
        episode_events=episode_events,
        latency_events=latency_events,
        normalize_actions=normalize_actions,
        random_episode_starts=True,
        seed=seed,
    )
    n_steps = min(128, max(2, episode_events // 2))
    batch_size = min(64, n_steps)
    model = ppo_class(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": AttnLOBFeatureExtractor,
            "features_extractor_kwargs": {
                "lob_mode": lob_mode,
                "use_dynamic_state": use_dynamic_state,
                "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint else None,
                "freeze_encoder": freeze_encoder,
            },
            "log_std_init": policy_log_std_init,
        },
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        learning_rate=1e-4,
        gamma=0.99,
        seed=seed,
        device=device,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(output_dir / "c_ppo_model")
    pl.DataFrame(
        [
            {
                "total_timesteps": total_timesteps,
                "train_events": train_dataset.orderbook.height,
                "episode_events": episode_events,
                "latency_events": latency_events,
                "lob_mode": lob_mode,
                "use_dynamic_state": use_dynamic_state,
                "normalize_actions": normalize_actions,
                "policy_log_std_init": policy_log_std_init,
                "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint else None,
                "freeze_encoder": freeze_encoder,
                "random_episode_starts": True,
            }
        ]
    ).write_csv(output_dir / "c_ppo_train_config.csv")
    return model


def _evaluate_ppo_on_test_episodes(
    model: object,
    dataset: LobDataset,
    *,
    episode_events: int,
    latency_events: int,
    normalize_actions: bool,
    seed: int,
) -> list[tuple[dict[str, float | int | str], list[dict[str, float | int]]]]:
    rows: list[tuple[dict[str, float | int | str], list[dict[str, float | int]]]] = []
    evaluation_events = min(episode_events, dataset.orderbook.height - 1)
    env = PaperMarketMakingEnv(
        dataset,
        episode_events=evaluation_events,
        latency_events=latency_events,
        normalize_actions=normalize_actions,
        seed=seed,
    )
    starts = _episode_starts(
        dataset,
        episode_events=evaluation_events,
        latency_events=latency_events,
    )
    for episode_id, episode_start in enumerate(starts):
        metrics, log_rows = _evaluate_ppo_model(
            model,
            env,
            seed=seed + episode_id,
            episode_start=episode_start,
        )
        metrics.update(
            {
                "episode_id": episode_id,
                "episode_start": episode_start,
                "episode_events": evaluation_events,
            }
        )
        for row in log_rows:
            row["episode_id"] = episode_id
            row["episode_start"] = episode_start
        rows.append((metrics, log_rows))
    return rows


def _train_ddqn_model(
    train_dataset: LobDataset,
    *,
    output_dir: Path,
    episode_events: int,
    latency_events: int,
    total_timesteps: int,
    seed: int,
    device: str,
    lob_mode: str = "attn",
    use_dynamic_state: bool = True,
    encoder_checkpoint: Path | None = None,
    freeze_encoder: bool = False,
) -> object:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = PaperDiscreteMarketMakingEnv(
        train_dataset,
        episode_events=episode_events,
        latency_events=latency_events,
        random_episode_starts=True,
        seed=seed,
    )
    config = DuelingDQNConfig(
        total_timesteps=total_timesteps,
        learning_starts=min(100, max(1, total_timesteps // 4)),
        buffer_size=max(1_000, total_timesteps * 4),
        batch_size=min(32, max(1, total_timesteps // 4)),
        target_update_interval=max(50, total_timesteps // 2),
        seed=seed,
    )
    model, train_result = train_dueling_dqn(
        env,
        config=config,
        lob_mode=lob_mode,
        use_dynamic_state=use_dynamic_state,
        encoder_checkpoint=str(encoder_checkpoint) if encoder_checkpoint else None,
        freeze_encoder=freeze_encoder,
        device=device,
    )
    save_dueling_dqn(model, output_dir / "d_dqn_model.pt", config=config, train_result=train_result)
    pl.DataFrame({"loss": train_result.losses}).write_csv(output_dir / "d_dqn_losses.csv")
    pl.DataFrame(
        [
            {
                "total_timesteps": total_timesteps,
                "updates": train_result.updates,
                "final_epsilon": train_result.final_epsilon,
                "train_events": train_dataset.orderbook.height,
                "episode_events": episode_events,
                "latency_events": latency_events,
                "lob_mode": lob_mode,
                "use_dynamic_state": use_dynamic_state,
                "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint else None,
                "freeze_encoder": freeze_encoder,
                "random_episode_starts": True,
            }
        ]
    ).write_csv(output_dir / "d_dqn_train_config.csv")
    return model


def _evaluate_ddqn_on_test_episodes(
    model,
    dataset: LobDataset,
    *,
    episode_events: int,
    latency_events: int,
    seed: int,
    device: str,
) -> list[tuple[dict[str, float | int | str], list[dict[str, float | int]]]]:
    rows: list[tuple[dict[str, float | int | str], list[dict[str, float | int]]]] = []
    evaluation_events = min(episode_events, dataset.orderbook.height - 1)
    env = PaperDiscreteMarketMakingEnv(
        dataset,
        episode_events=evaluation_events,
        latency_events=latency_events,
        seed=seed,
    )
    starts = _episode_starts(
        dataset,
        episode_events=evaluation_events,
        latency_events=latency_events,
    )
    for episode_id, episode_start in enumerate(starts):
        metrics, log_rows = evaluate_dueling_dqn(
            model,
            env,
            seed=seed + episode_id,
            device=device,
            episode_start=episode_start,
        )
        metrics.update(
            {
                "episode_id": episode_id,
                "episode_start": episode_start,
                "episode_events": evaluation_events,
            }
        )
        for row in log_rows:
            row["episode_id"] = episode_id
            row["episode_start"] = episode_start
        rows.append((metrics, log_rows))
    return rows


def _run_latency_synthetic_table(
    *,
    train_dataset: LobDataset,
    test_dataset: LobDataset,
    output_dir: Path,
    latencies: list[int],
    episode_events: int,
    tabular_episodes: int,
    agent_timesteps: int,
    encoder_checkpoint: Path,
    seed: int,
    device: str,
    ppo_class,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, float | int | str | bool]] = []
    trade_rows: list[dict[str, float | int | str]] = []
    baseline_strategies = _baseline_strategies_for_train_data(
        train_dataset,
        episode_events=episode_events,
        tabular_episodes=tabular_episodes,
        seed=seed + 50_000,
    )
    ppo_model = _train_ppo_model(
        train_dataset,
        output_dir=output_dir / "latency_train" / "c_ppo",
        episode_events=episode_events,
        latency_events=1,
        total_timesteps=agent_timesteps,
        seed=seed + 51_000,
        device=device,
        ppo_class=ppo_class,
        encoder_checkpoint=encoder_checkpoint,
        normalize_actions=True,
        policy_log_std_init=FULL_REPLICATION_PPO_LOG_STD_INIT,
    )
    ddqn_model = _train_ddqn_model(
        train_dataset,
        output_dir=output_dir / "latency_train" / "d_dqn",
        episode_events=episode_events,
        latency_events=1,
        total_timesteps=agent_timesteps,
        seed=seed + 52_000,
        device=device,
        encoder_checkpoint=encoder_checkpoint,
    )

    for latency_index, latency in enumerate(latencies):
        for row, log_rows in _evaluate_strategies_on_test_episodes(
            test_dataset,
            strategies=baseline_strategies,
            episode_events=episode_events,
            latency_events=latency,
            seed=seed + latency_index,
        ):
            row.update(
                {
                    "stock": test_dataset.stock,
                    "day": test_dataset.day,
                    "latency_events": latency,
                }
            )
            metrics_rows.append(row)
            _extend_tagged_trades(
                trade_rows,
                log_rows,
                stock=test_dataset.stock,
                day=test_dataset.day,
                method=str(row["method"]),
                latency_events=latency,
            )

        for row, log_rows in _evaluate_ppo_on_test_episodes(
            ppo_model,
            test_dataset,
            episode_events=episode_events,
            latency_events=latency,
            normalize_actions=True,
            seed=seed + 60_000 + latency_index,
        ):
            row.update(
                {
                    "method": "C-PPO",
                    "stock": test_dataset.stock,
                    "day": test_dataset.day,
                    "latency_events": latency,
                    "trained_latency_events": 1,
                    "total_timesteps": agent_timesteps,
                    "normalize_actions": True,
                    "policy_log_std_init": FULL_REPLICATION_PPO_LOG_STD_INIT,
                    "encoder_checkpoint": str(encoder_checkpoint),
                }
            )
            metrics_rows.append(row)
            _extend_tagged_trades(
                trade_rows,
                log_rows,
                stock=test_dataset.stock,
                day=test_dataset.day,
                method="C-PPO",
                latency_events=latency,
            )

        for row, log_rows in _evaluate_ddqn_on_test_episodes(
            ddqn_model,
            test_dataset,
            episode_events=episode_events,
            latency_events=latency,
            seed=seed + 70_000 + latency_index,
            device=device,
        ):
            row.update(
                {
                    "method": "D-DQN",
                    "stock": test_dataset.stock,
                    "day": test_dataset.day,
                    "latency_events": latency,
                    "trained_latency_events": 1,
                    "total_timesteps": agent_timesteps,
                    "encoder_checkpoint": str(encoder_checkpoint),
                }
            )
            metrics_rows.append(row)
            _extend_tagged_trades(
                trade_rows,
                log_rows,
                stock=test_dataset.stock,
                day=test_dataset.day,
                method="D-DQN",
                latency_events=latency,
            )
    return metrics_rows, trade_rows


def _run_ablation_synthetic_table(
    *,
    train_dataset: LobDataset,
    test_dataset: LobDataset,
    output_dir: Path,
    episode_events: int,
    agent_timesteps: int,
    encoder_checkpoint: Path,
    seed: int,
    device: str,
    ppo_class,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = (
        ("full", "attn", True),
        ("without_lob", "none", True),
        ("without_attn_lob", "mlp", True),
        ("without_dynamic", "attn", False),
    )
    metrics_rows: list[dict[str, float | int | str | bool]] = []
    trade_rows: list[dict[str, float | int | str]] = []
    for variant_index, (variant, lob_mode, use_dynamic_state) in enumerate(variants):
        variant_checkpoint = encoder_checkpoint if lob_mode == "attn" else None
        ppo_model = _train_ppo_model(
            train_dataset,
            output_dir=output_dir / "c_ppo" / variant,
            episode_events=episode_events,
            latency_events=1,
            total_timesteps=agent_timesteps,
            seed=seed + 70_000 + variant_index,
            device=device,
            ppo_class=ppo_class,
            lob_mode=lob_mode,
            use_dynamic_state=use_dynamic_state,
            encoder_checkpoint=variant_checkpoint,
            normalize_actions=True,
            policy_log_std_init=FULL_REPLICATION_PPO_LOG_STD_INIT,
        )
        for row, log_rows in _evaluate_ppo_on_test_episodes(
            ppo_model,
            test_dataset,
            episode_events=episode_events,
            latency_events=1,
            normalize_actions=True,
            seed=seed + 75_000 + variant_index,
        ):
            row.update(
                {
                    "method": "C-PPO",
                    "variant": variant,
                    "stock": test_dataset.stock,
                    "day": test_dataset.day,
                    "total_timesteps": agent_timesteps,
                    "latency_events": 1,
                    "lob_mode": lob_mode,
                    "use_dynamic_state": use_dynamic_state,
                    "normalize_actions": True,
                    "policy_log_std_init": FULL_REPLICATION_PPO_LOG_STD_INIT,
                    "encoder_checkpoint": str(variant_checkpoint) if variant_checkpoint else None,
                }
            )
            metrics_rows.append(row)
            _extend_tagged_trades(
                trade_rows,
                log_rows,
                stock=test_dataset.stock,
                day=test_dataset.day,
                method="C-PPO",
                variant=variant,
            )

        ddqn_model = _train_ddqn_model(
            train_dataset,
            output_dir=output_dir / "d_dqn" / variant,
            episode_events=episode_events,
            latency_events=1,
            total_timesteps=agent_timesteps,
            seed=seed + 80_000 + variant_index,
            device=device,
            lob_mode=lob_mode,
            use_dynamic_state=use_dynamic_state,
            encoder_checkpoint=variant_checkpoint,
        )
        for row, log_rows in _evaluate_ddqn_on_test_episodes(
            ddqn_model,
            test_dataset,
            episode_events=episode_events,
            latency_events=1,
            seed=seed + 85_000 + variant_index,
            device=device,
        ):
            row.update(
                {
                    "method": "D-DQN",
                    "variant": variant,
                    "stock": test_dataset.stock,
                    "day": test_dataset.day,
                    "total_timesteps": agent_timesteps,
                    "latency_events": 1,
                    "lob_mode": lob_mode,
                    "use_dynamic_state": use_dynamic_state,
                    "encoder_checkpoint": str(variant_checkpoint) if variant_checkpoint else None,
                }
            )
            metrics_rows.append(row)
            _extend_tagged_trades(
                trade_rows,
                log_rows,
                stock=test_dataset.stock,
                day=test_dataset.day,
                method="D-DQN",
                variant=variant,
            )
    return metrics_rows, trade_rows


def _extend_tagged_trades(
    target: list[dict[str, float | int | str]],
    rows: list[dict[str, float | int | str]],
    *,
    stock: str,
    day: str,
    method: str | None = None,
    latency_events: int | None = None,
    variant: str | None = None,
) -> None:
    for row in rows:
        tagged = dict(row)
        tagged["stock"] = stock
        tagged["day"] = day
        if method is not None:
            tagged["method"] = method
        if latency_events is not None:
            tagged["latency_events"] = latency_events
        if variant is not None:
            tagged["variant"] = variant
        target.append(tagged)


def _plot_attention_from_ppo_encoder(model: object, dataset: LobDataset, output_path: Path) -> None:
    index = min(max(PAPER.window_length - 1, 80), dataset.orderbook.height - 1)
    start = index - PAPER.window_length + 1
    lob_values = dataset.orderbook.select(lob_columns()).slice(start, PAPER.window_length)
    window = normalize_lob_window(lob_values.to_numpy())
    policy = getattr(model, "policy")
    extractor = getattr(policy, "features_extractor")
    encoder = getattr(extractor, "lob_encoder")
    encoder.eval()
    device = next(encoder.parameters()).device
    with torch.no_grad():
        _, weights = encoder(
            torch.from_numpy(window).float().unsqueeze(0).to(device),
            return_attention_weights=True,
        )
    plot_attention_heatmap(weights.squeeze(0).cpu().numpy(), output_path, lob_window=window)


def _write_full_replication_config(
    path: Path,
    *,
    stock_specs: list[tuple[str, float]],
    train_days: int,
    test_days: int,
    events_per_day: int,
    episode_events: int,
    pretrain_events: int,
    pretrain_epochs: int,
    agent_timesteps: int,
    tabular_episodes: int,
    latencies: list[int],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full Synthetic Replication Config",
        "",
        f"- stocks: {', '.join(stock for stock, _ in stock_specs)}",
        f"- base_prices: {', '.join(str(price) for _, price in stock_specs)}",
        f"- train_days: {train_days}",
        f"- test_days: {test_days}",
        (
            f"- train_date_range: {PAPER_TRADING_DAYS_201911[0]}.."
            f"{PAPER_TRADING_DAYS_201911[train_days - 1]}"
        ),
        (
            f"- test_date_range: {PAPER_TRADING_DAYS_201911[train_days]}.."
            f"{PAPER_TRADING_DAYS_201911[train_days + test_days - 1]}"
        ),
        f"- events_per_day: {events_per_day}",
        f"- episode_events: {episode_events}",
        f"- pretrain_events: {pretrain_events}",
        f"- pretrain_epochs: {pretrain_epochs}",
        f"- agent_timesteps: {agent_timesteps}",
        f"- tabular_episodes: {tabular_episodes}",
        f"- latency_events: {', '.join(str(value) for value in latencies)}",
        f"- seed: {seed}",
        "",
        "Synthetic replay data is the only planned data source for this command.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_full_replication_index(path: Path, *, paths: list[Path]) -> None:
    lines = ["# Full Synthetic Replication Artifacts", ""]
    for artifact in paths:
        lines.append(f"- `{artifact.relative_to(path.parent)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_synthetic_pretrain(
    *,
    model_name: str,
    output_dir: Path,
    events: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    tracker: WandbTracker | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=events, seed=seed))
    input_shape = pretrain_input_shape(model_name)
    arrays = build_pretrain_arrays(dataset, window_length=input_shape[0])
    model = make_pretrain_model(model_name)
    metrics = train_lob_classifier(
        model,
        arrays,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    safe_model_name = model_name.lower().replace("-", "_")
    implementation_param = count_parameters(model)
    implementation_encoder_param = count_encoder_parameters(model)
    paper_param = paper_reported_parameter_count(model_name)
    row = {
        "model": model_name,
        **metrics.__dict__,
        "input_window_length": input_shape[0],
        "implementation_param": implementation_param,
        "implementation_encoder_param": implementation_encoder_param,
        "paper_reported_param": paper_param,
        "param_matches_paper_report": implementation_encoder_param == paper_param,
        "full_param_matches_paper_report": implementation_param == paper_param,
        "encoder_param_matches_paper_report": implementation_encoder_param == paper_param,
    }
    metrics_path = output_dir / f"{safe_model_name}_pretrain_metrics.csv"
    model_path = output_dir / f"{safe_model_name}_pretrain_model.pt"
    pl.DataFrame([row]).write_csv(metrics_path)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"model": model_name, "state_dict": state_dict}, model_path)
    if tracker is not None:
        tracker.log_metrics(row, prefix="pretrain")
        tracker.update_summary(row, prefix="pretrain")
        tracker.log_artifact(
            [metrics_path, model_path],
            name=f"{safe_model_name}-pretrain-seed-{seed}",
            artifact_type="pretrain",
        )
    console.print(f"[green]wrote[/green] {metrics_path}")
    console.print(f"[green]wrote[/green] {model_path}")


def _evaluate_ppo_model(
    model: object, env: PaperMarketMakingEnv, *, seed: int, episode_start: int = 0
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    obs, _ = env.reset(seed=seed, options={"episode_start": episode_start})
    done = False
    info: dict[str, object] = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # type: ignore[attr-defined]
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    metrics = info.get("metrics", {})
    trade_log = info.get("trade_log", [])
    if not isinstance(metrics, dict) or not isinstance(trade_log, list):
        raise RuntimeError("PPO evaluation did not return terminal metrics")
    return metrics, trade_log


def _parse_int_list(spec: str) -> list[int]:
    values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not values:
        raise typer.BadParameter("expected at least one integer")
    if min(values) < 1:
        raise typer.BadParameter("latencies must be positive integers")
    return values


def _read_table(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".csv":
        return _read_metrics_csv(path)
    raise typer.BadParameter("table path must end in .parquet or .csv")


def _read_metrics_csv(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(path, schema_overrides={"stock": pl.Utf8})
    except TypeError:
        return pl.read_csv(path)
