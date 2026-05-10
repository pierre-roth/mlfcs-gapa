from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import DiagnosticsConfig
from .models import PPOActorCritic
from .paper_env import PaperTradingEnv
from .paper_policies import ContinuousActionPolicy
from .real_data import load_market_days
from .training import _configure_torch, _episode_iter, _trading_backbone


def export_ppo_attention(
    checkpoint: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
    max_episodes: int = 4,
    max_steps_per_episode: int = 200,
) -> dict[str, Any]:
    """Export Attn-LOB temporal attention weights for a deterministic PPO policy."""

    _configure_torch(device)
    payload = torch.load(checkpoint, map_location=device)
    config = _config_from_checkpoint(payload.get("config", {}))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PPOActorCritic(
        _trading_backbone(config, None, device),
        initial_log_std=config.ppo_initial_log_std,
        initial_spread_bias=config.ppo_initial_spread_bias,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    days = load_market_days(config, "test")
    rows: list[dict[str, Any]] = []
    episode_count = 0
    with torch.no_grad():
        for day in days:
            for episode_index, start, stop in _episode_iter(day, config, config.max_eval_episodes_per_day):
                if episode_count >= max_episodes:
                    break
                env = PaperTradingEnv(day, config, start, stop, episode_index=episode_index, rng_seed=config.seed)
                state = env.reset()
                terminal = False
                step = 0
                while not terminal and step < max_steps_per_episode:
                    lob = torch.from_numpy(state.lob_state).unsqueeze(0).to(device=device, dtype=torch.float32)
                    market = torch.from_numpy(state.market_state).unsqueeze(0).to(device=device, dtype=torch.float32)
                    agent = torch.from_numpy(state.agent_state).unsqueeze(0).to(device=device, dtype=torch.float32)
                    mean, _, value = model(lob, market, agent)
                    weights = _attention_weights(model, lob).detach().cpu().numpy()[0]
                    action = mean.detach().cpu().numpy()[0]
                    for head_idx, head_weights in enumerate(weights):
                        for lookback_pos, weight in enumerate(head_weights):
                            rows.append(
                                {
                                    "day": day.day,
                                    "episode_index": episode_index,
                                    "step": step,
                                    "event_idx": int(env.event_idx),
                                    "head": head_idx,
                                    "lookback_pos": lookback_pos,
                                    "lookback_age": config.lookback - 1 - lookback_pos,
                                    "attention": float(weight),
                                    "action_bias": float(action[0]),
                                    "action_spread": float(action[1]),
                                    "value": float(value.detach().cpu().item()),
                                    "inventory": int(env.inventory),
                                }
                            )
                    result = env.step(ContinuousActionPolicy(action).act(state, env))
                    terminal = result.terminal
                    if result.state is not None:
                        state = result.state
                    step += 1
                episode_count += 1
            if episode_count >= max_episodes:
                break

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ppo_attention_steps.csv", index=False)
    summary = _summarize_attention(frame)
    (output_dir / "ppo_attention_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    if not frame.empty:
        _plot_attention(frame, output_dir)
    return summary


def _config_from_checkpoint(payload: dict[str, Any]) -> DiagnosticsConfig:
    valid = {field.name for field in fields(DiagnosticsConfig)}
    values = {key: value for key, value in payload.items() if key in valid}
    config = DiagnosticsConfig(**values)
    config.apply_mode_defaults()
    return config


def _attention_weights(model: PPOActorCritic, lob_state: torch.Tensor) -> torch.Tensor:
    encoder = model.backbone.encoder
    if not all(hasattr(encoder, name) for name in ("spatial", "inception_3", "inception_5", "inception_pool", "query", "key", "attention_key_dim")):
        raise TypeError("The PPO checkpoint does not use an Attn-LOB encoder with temporal attention.")
    x = lob_state.permute(0, 3, 1, 2).contiguous()
    x = encoder.spatial(x)
    x = torch.cat([encoder.inception_3(x), encoder.inception_5(x), encoder.inception_pool(x)], dim=1)
    x = x.squeeze(-1).permute(0, 2, 1).contiguous()
    query = x[:, -1:, :]
    q = encoder._split_heads(encoder.query(query))
    k = encoder._split_heads(encoder.key(x))
    scores = torch.matmul(q, k.transpose(-2, -1)) / (encoder.attention_key_dim**0.5)
    return torch.softmax(scores, dim=-1).squeeze(2)


def _summarize_attention(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    by_age = frame.groupby("lookback_age")["attention"].mean().sort_values(ascending=False)
    by_head = frame.groupby("head")["attention"].agg(["mean", "std"]).reset_index()
    return {
        "rows": int(len(frame)),
        "episodes": int(frame[["day", "episode_index"]].drop_duplicates().shape[0]),
        "steps": int(frame[["day", "episode_index", "step"]].drop_duplicates().shape[0]),
        "heads": int(frame["head"].nunique()),
        "lookback": int(frame["lookback_pos"].nunique()),
        "top_attention_ages": [{"lookback_age": int(idx), "attention": float(value)} for idx, value in by_age.head(10).items()],
        "head_summary": [
            {"head": int(row["head"]), "mean": float(row["mean"]), "std": float(row["std"])}
            for _, row in by_head.iterrows()
        ],
    }


def _plot_attention(frame: pd.DataFrame, output_dir: Path) -> None:
    aggregate = frame.pivot_table(index="head", columns="lookback_age", values="attention", aggfunc="mean")
    aggregate = aggregate.reindex(sorted(aggregate.columns, reverse=True), axis=1)
    _heatmap(aggregate, output_dir / "ppo_attention_aggregate.png", "Mean PPO Attn-LOB attention")

    for (day, episode_index), group in frame.groupby(["day", "episode_index"]):
        episode = group.pivot_table(index="head", columns="lookback_age", values="attention", aggfunc="mean")
        episode = episode.reindex(sorted(episode.columns, reverse=True), axis=1)
        _heatmap(episode, output_dir / f"ppo_attention_{day}_episode{int(episode_index):02d}.png", f"{day} episode {episode_index}")


def _heatmap(data: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    image = ax.imshow(data.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Lookback age, newer to older")
    ax.set_ylabel("Attention head")
    if len(data.columns):
        ticks = np.linspace(0, len(data.columns) - 1, min(8, len(data.columns)), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(data.columns[idx]) for idx in ticks])
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(idx) for idx in data.index])
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PPO Attn-LOB attention diagnostics.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-episodes", type=int, default=4)
    parser.add_argument("--max-steps-per-episode", type=int, default=200)
    args = parser.parse_args()
    summary = export_ppo_attention(
        args.checkpoint,
        args.output_dir,
        device=args.device,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
