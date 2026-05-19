"""Command-line entrypoints for replication utilities."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import polars as pl
import torch
import typer
from rich.console import Console

from mlfcs_gapa.data.features import normalize_lob_window
from mlfcs_gapa.data.io import write_lob_dataset
from mlfcs_gapa.data.lobster import load_lobster_csv
from mlfcs_gapa.data.pretraining import build_pretrain_arrays
from mlfcs_gapa.data.schema import lob_columns
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
)
from mlfcs_gapa.env.gym_env import PaperMarketMakingEnv
from mlfcs_gapa.experiments.figures import (
    plot_attention_heatmap,
    plot_decision_trace,
    plot_latency_figure,
)
from mlfcs_gapa.experiments.reports import summarize_paper_table
from mlfcs_gapa.models.attn_lob import AttnLOBClassifier
from mlfcs_gapa.models.pretrain_models import make_pretrain_model
from mlfcs_gapa.paper.constants import PAPER, PAPER_TRADING_DAYS_201911
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


@app.callback()
def main() -> None:
    """Utilities for the paper-faithful LOB market-making replication."""


@app.command("generate-synthetic")
def generate_synthetic(
    output_dir: Path = typer.Option(Path("data/synthetic"), help="Output root for Parquet files."),
    stock: str = typer.Option("000001", help="Synthetic stock code."),
    days: int = typer.Option(1, min=1, max=len(PAPER_TRADING_DAYS_201911), help="Number of days."),
    events_per_day: int = typer.Option(6_000, min=100, help="Events generated per day."),
    base_price: float = typer.Option(16.45, min=0.01, help="Starting stock price."),
    seed: int = typer.Option(1, help="Base random seed."),
) -> None:
    """Generate paper-shaped synthetic LOB data.

    The output is source-separated under ``data/synthetic`` by default and uses
    the same canonical schema as future real-data adapters.
    """

    for day_index, day in enumerate(PAPER_TRADING_DAYS_201911[:days]):
        config = SyntheticLobConfig(
            stock=stock,
            day=day,
            n_events=events_per_day,
            base_price=base_price,
            seed=seed + day_index,
        )
        dataset = generate_synthetic_lob_day(config)
        written = write_lob_dataset(dataset, output_dir)
        console.print(
            f"[green]wrote[/green] {written} "
            f"({dataset.orderbook.height:,} events, {len(dataset.orderbook.columns)} orderbook columns)"
        )


@app.command("convert-lobster")
def convert_lobster(
    message_path: Path = typer.Argument(..., help="LOBSTER message CSV."),
    orderbook_path: Path = typer.Argument(..., help="LOBSTER orderbook CSV."),
    output_dir: Path = typer.Option(Path("data/lobster"), help="Output root for Parquet files."),
    stock: str = typer.Option(..., help="Ticker or stock code."),
    day: str = typer.Option(..., help="Trading day as YYYY-MM-DD."),
    levels: int = typer.Option(PAPER.lob_levels, min=PAPER.lob_levels, help="LOBSTER levels."),
    price_scale: float = typer.Option(10_000.0, min=1.0, help="Fixed-point price scale."),
) -> None:
    """Convert LOBSTER CSV files to the canonical replication schema."""

    dataset = load_lobster_csv(
        message_path=message_path,
        orderbook_path=orderbook_path,
        stock=stock,
        day=day,
        levels=levels,
        price_scale=price_scale,
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
) -> None:
    """Run paper baseline strategies on synthetic data."""

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
) -> None:
    """Run a Figure-2-style latency sweep for synthetic baselines."""

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
) -> None:
    """Run a small Attn-LOB pretraining experiment on synthetic data."""

    _run_synthetic_pretrain(
        model_name="Attn-LOB",
        output_dir=output_dir,
        events=events,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        seed=seed,
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
) -> None:
    """Run one Table I pretraining model on synthetic data."""

    _run_synthetic_pretrain(
        model_name=model_name,
        output_dir=output_dir,
        events=events,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        seed=seed,
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
    learning_rate: float = typer.Option(3e-4, min=1e-8, help="PPO learning rate."),
    encoder_checkpoint: Path | None = typer.Option(None, help="Optional Attn-LOB checkpoint."),
    freeze_encoder: bool = typer.Option(False, help="Freeze loaded Attn-LOB encoder weights."),
    lob_mode: str = typer.Option("attn", help="LOB feature mode: attn, mlp, or none."),
    use_dynamic_state: bool = typer.Option(True, help="Include the 24-dimensional dynamic state."),
    use_agent_state: bool = typer.Option(True, help="Include inventory/time agent state."),
    device: str = typer.Option(
        "auto", help="Torch device for Stable-Baselines3, e.g. auto/cpu/cuda."
    ),
    seed: int = typer.Option(1, help="Random seed."),
) -> None:
    """Train a paper C-PPO smoke/experiment run on synthetic data."""

    if batch_size > n_steps:
        raise typer.BadParameter("batch-size must be <= n-steps for the single-env PPO runner")

    from stable_baselines3 import PPO

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=events, seed=seed))
    env = PaperMarketMakingEnv(
        dataset,
        episode_events=min(episode_events, events - 1),
        latency_events=latency_events,
        seed=seed,
    )
    policy_kwargs = {
        "features_extractor_class": AttnLOBFeatureExtractor,
        "features_extractor_kwargs": {
            "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint else None,
            "freeze_encoder": freeze_encoder,
            "lob_mode": lob_mode,
            "use_dynamic_state": use_dynamic_state,
            "use_agent_state": use_agent_state,
        },
    }
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
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
            "events": events,
            "episode_events": min(episode_events, events - 1),
            "latency_events": latency_events,
            "lob_mode": lob_mode,
            "use_dynamic_state": use_dynamic_state,
            "use_agent_state": use_agent_state,
        }
    )
    metrics_path = output_dir / "c_ppo_metrics.csv"
    trades_path = output_dir / "c_ppo_trades.parquet"
    pl.DataFrame([metrics]).write_csv(metrics_path)
    pl.DataFrame(trade_log).write_parquet(trades_path)
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
    use_agent_state: bool = typer.Option(True, help="Include inventory/time agent state."),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda."),
    seed: int = typer.Option(1, help="Random seed."),
) -> None:
    """Train the paper's discrete dueling Double DQN agent on synthetic data."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=events, seed=seed))
    env = PaperDiscreteMarketMakingEnv(
        dataset,
        episode_events=min(episode_events, events - 1),
        latency_events=latency_events,
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
        use_agent_state=use_agent_state,
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
            "use_agent_state": use_agent_state,
        }
    )
    metrics_path = output_dir / "d_dqn_metrics.csv"
    trades_path = output_dir / "d_dqn_trades.parquet"
    losses_path = output_dir / "d_dqn_losses.csv"
    pl.DataFrame([metrics]).write_csv(metrics_path)
    pl.DataFrame(trade_log).write_parquet(trades_path)
    pl.DataFrame({"loss": train_result.losses}).write_csv(losses_path)
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
) -> None:
    """Measure Table-III-style runtime on a synthetic smoke workload."""

    from stable_baselines3 import PPO

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
    console.print(f"[green]wrote[/green] {output_path}")


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
    plot_attention_heatmap(weights.squeeze(0).numpy(), output_path)
    console.print(f"[green]wrote[/green] {output_path}")


def _run_synthetic_pretrain(
    *,
    model_name: str,
    output_dir: Path,
    events: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=events, seed=seed))
    arrays = build_pretrain_arrays(dataset)
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
    row = {"model": model_name, **metrics.__dict__}
    metrics_path = output_dir / f"{safe_model_name}_pretrain_metrics.csv"
    model_path = output_dir / f"{safe_model_name}_pretrain_model.pt"
    pl.DataFrame([row]).write_csv(metrics_path)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"model": model_name, "state_dict": state_dict}, model_path)
    console.print(f"[green]wrote[/green] {metrics_path}")
    console.print(f"[green]wrote[/green] {model_path}")


def _evaluate_ppo_model(
    model: object, env: PaperMarketMakingEnv, *, seed: int
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    obs, _ = env.reset(seed=seed)
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
