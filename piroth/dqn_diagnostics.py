from __future__ import annotations

import argparse
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import numpy as np
import pandas as pd
import torch

from .config import DiagnosticsConfig
from .models import DuelingDQN
from .paper_env import PaperTradingEnv
from .paper_policies import DiscreteActionPolicy
from .real_data import load_market_days
from .training import _episode_iter, _state_tensors, _trading_backbone

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def run_dqn_action_diagnostics(
    config: DiagnosticsConfig,
    checkpoint: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
    force_liquidate_when_breached: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay deterministic DQN actions and summarize action/inventory behavior."""
    if device.startswith("cuda") and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = DuelingDQN(_trading_backbone(config, None, device)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    model.eval()

    days = load_market_days(config, "test")
    step_rows: list[dict[str, float | int | str | bool]] = []
    episode_rows: list[dict[str, float | int | str]] = []
    max_inventory = config.max_inventory_units * config.trade_unit

    for day in days:
        for episode_index, start, stop in _episode_iter(day, config, config.max_eval_episodes_per_day):
            env = PaperTradingEnv(day, config, start, stop, episode_index=episode_index, rng_seed=config.seed)
            state = env.reset()
            terminal = False
            steps = 0
            while not terminal:
                inventory_before = env.inventory
                lob, market, agent = _state_tensors(state, device)
                with torch.no_grad():
                    q_values = model(lob, market, agent).detach().cpu().numpy()[0]
                model_action = int(np.argmax(q_values))
                action = model_action
                breach_before = abs(inventory_before) >= max_inventory
                if force_liquidate_when_breached and breach_before:
                    action = 7
                policy_action = DiscreteActionPolicy(action).act(state, env)
                result = env.step(policy_action)
                inventory_after = env.inventory
                breach_after = abs(inventory_after) >= max_inventory
                step_rows.append(
                    {
                        "day": day.day,
                        "episode_index": episode_index,
                        "step": steps,
                        "event_idx": int(result.info["event_idx"]),
                        "action": action,
                        "model_action": model_action,
                        "forced_action": action != model_action,
                        "inventory_before": inventory_before,
                        "inventory_after": inventory_after,
                        "breach_before": breach_before,
                        "breach_after": breach_after,
                        "liquidation_action": action == 7,
                        "selected_q": float(q_values[action]),
                        "liquidation_q": float(q_values[7]),
                        "q_margin_liquidation_vs_selected": float(q_values[7] - q_values[action]),
                        "trade_volume": int(result.info["trade_volume"]),
                        "reward": result.reward,
                    }
                )
                terminal = result.terminal
                if result.state is not None:
                    state = result.state
                steps += 1
            metrics = env.metrics()
            episode_rows.append(
                {
                    "day": metrics.day,
                    "episode_index": metrics.episode_index,
                    "pnl": metrics.pnl,
                    "reward": metrics.reward,
                    "avg_abs_position": metrics.avg_abs_position,
                    "avg_spread": metrics.avg_spread,
                    "fill_rate": metrics.fill_rate,
                    "turnover": metrics.turnover,
                    "trades": metrics.trades,
                }
            )

    steps = pd.DataFrame(step_rows)
    episodes = pd.DataFrame(episode_rows)
    steps.to_csv(output_dir / "dqn_action_steps.csv", index=False)
    episodes.to_csv(output_dir / "dqn_action_episodes.csv", index=False)
    _summaries(steps, episodes, output_dir, lot_size=config.trade_unit)
    return steps, episodes


def _summaries(steps: pd.DataFrame, episodes: pd.DataFrame, output_dir: Path, *, lot_size: int) -> None:
    action_counts = steps["action"].value_counts().rename_axis("action").reset_index(name="count")
    action_counts["fraction"] = action_counts["count"] / max(len(steps), 1)
    action_counts.to_csv(output_dir / "dqn_action_counts.csv", index=False)

    by_breach = (
        steps.groupby("breach_before")
        .agg(
            steps=("action", "size"),
            liquidation_rate=("liquidation_action", "mean"),
            mean_abs_inventory=("inventory_before", lambda x: np.mean(np.abs(x))),
            mean_q_margin_liquidation_vs_selected=("q_margin_liquidation_vs_selected", "mean"),
        )
        .reset_index()
    )
    by_breach.to_csv(output_dir / "dqn_breach_summary.csv", index=False)

    steps = steps.copy()
    steps["inventory_bucket"] = pd.cut(
        steps["inventory_before"] / lot_size,
        bins=[-np.inf, -10, -5, 0, 5, 10, np.inf],
        labels=["<-10", "-10..-5", "-5..0", "0..5", "5..10", ">10"],
        include_lowest=True,
    )
    bucket_summary = (
        steps.groupby("inventory_bucket", observed=True)
        .agg(
            steps=("action", "size"),
            liquidation_rate=("liquidation_action", "mean"),
            mean_q_margin_liquidation_vs_selected=("q_margin_liquidation_vs_selected", "mean"),
            mean_abs_inventory=("inventory_before", lambda x: np.mean(np.abs(x))),
        )
        .reset_index()
    )
    bucket_summary.to_csv(output_dir / "dqn_inventory_bucket_summary.csv", index=False)

    mean_metrics = episodes.mean(numeric_only=True).to_frame("mean").reset_index().rename(columns={"index": "metric"})
    mean_metrics.to_csv(output_dir / "dqn_diagnostic_episode_means.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN action/inventory diagnostics for paper replication runs.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-liquidate-when-breached", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args()
    config = DiagnosticsConfig(**_load_overrides(DiagnosticsConfig, args.set))
    config.apply_mode_defaults()
    run_dqn_action_diagnostics(
        config,
        args.checkpoint,
        args.output_dir,
        device=args.device,
        force_liquidate_when_breached=args.force_liquidate_when_breached,
    )


def _load_overrides(config_cls, set_items: list[str]) -> dict[str, Any]:
    by_name = {field.name: field for field in fields(config_cls)}
    type_hints = get_type_hints(config_cls)
    overrides: dict[str, Any] = {}
    for item in set_items:
        if "=" not in item:
            raise SystemExit(f"Invalid --set value {item!r}; expected FIELD=VALUE")
        key, raw = item.split("=", 1)
        if key not in by_name:
            raise SystemExit(f"Unknown config field: {key}")
        overrides[key] = _coerce(raw, type_hints.get(key, by_name[key].type), _field_default(by_name[key]))
    return overrides


def _field_default(field) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()
    return None


def _coerce(raw: str, annotation: Any, default: Any) -> Any:
    target, is_optional = _strip_optional(annotation)
    if is_optional and raw.strip().lower() in {"", "none", "null"}:
        return None
    origin = get_origin(target)
    if origin is list:
        item_type = get_args(target)[0] if get_args(target) else str
        return [_coerce(item.strip(), item_type, None) for item in raw.split(",") if item.strip()]
    if target is bool:
        return _parse_bool(raw)
    if target is int:
        return int(raw)
    if target is float:
        return float(raw)
    if target is str:
        return raw
    if isinstance(default, bool):
        return _parse_bool(raw)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    non_none = [arg for arg in args if arg is not type(None)]
    if non_none and len(non_none) != len(args):
        return non_none[0], True
    return annotation, False


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    raise ValueError(f"Cannot parse boolean value from {raw!r}")


if __name__ == "__main__":
    main()
