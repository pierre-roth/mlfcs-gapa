from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyrallis
import torch

from .config import TrainConfig
from .data import load_splits
from .env import ContinuousMarketEnv
from .models import ContinuousActorCritic, SharedStateEncoder, build_backbone
from .rl import train_ppo
from .utils import ensure_dir, save_json


def _build_model(config: TrainConfig, symbol: str):
    backbone = build_backbone(config.pretrain_backbone, config.lookback)
    ckpt = Path(config.output_dir()) / symbol / "pretrain" / config.backbone_name
    if ckpt.exists():
        backbone.load_state_dict(torch.load(ckpt, map_location="cpu"))
    for param in backbone.parameters():
        param.requires_grad = config.backbone_trainable
    encoder = SharedStateEncoder(backbone, flat_dim=48)
    return ContinuousActorCritic(encoder, action_dim=2)


def evaluate_model(envs, model, config: TrainConfig, method_name: str = "C_PPO"):
    model.to(config.device)
    model.eval()
    results = []
    with torch.no_grad():
        for env in envs:
            for episode_index, span in enumerate(env.selected_episodes(config.max_eval_episodes_per_day)):
                env.set_eval_context(episode_index)
                obs = env.reset(span)
                done = False
                while not done:
                    lob = torch.tensor(obs.lob[None, :, :], dtype=torch.float32, device=config.device)
                    flat = torch.tensor(obs.flat[None, :], dtype=torch.float32, device=config.device)
                    dist, _ = model.dist_value(lob, flat)
                    obs, _, done, _ = env.step(dist.mean.squeeze(0).cpu().numpy())
                results.append(env.episode_result(method_name, episode_index))
    return results


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    summary = {}
    for column in ["pnl", "nd_pnl", "pnl_map", "profit_ratio", "avg_abs_position", "avg_spread", "turnover", "reward", "trades", "fill_rate"]:
        if column in frame:
            summary[f"{column}_mean"] = float(frame[column].mean())
    if "pnl" in frame and frame["pnl"].std(ddof=0) > 0:
        summary["sharpe"] = float(frame["pnl"].mean() / frame["pnl"].std(ddof=0))
    else:
        summary["sharpe"] = 0.0
    return summary


def run_train(config: TrainConfig) -> dict[str, dict[str, float]]:
    config.apply_mode_defaults()
    summaries = {}
    for symbol in config.symbols:
        splits = load_splits(config, symbol)
        train_envs = [ContinuousMarketEnv(day, config) for day in splits["train"]]
        val_days = splits["val"] or splits["test"]
        val_envs = [ContinuousMarketEnv(day, config) for day in val_days]
        test_envs = [ContinuousMarketEnv(day, config) for day in splits["test"]]
        model = _build_model(config, symbol)
        symbol_dir = ensure_dir(Path(config.output_dir()) / symbol / "ppo")

        def _select(candidate, epoch):
            results = evaluate_model(val_envs, candidate, config, method_name="C_PPO_val")
            return summarize(pd.DataFrame(results))

        model, history, training_meta = train_ppo(train_envs, model, config, select_fn=_select if config.ppo_select_best_model else None)
        torch.save(model.state_dict(), symbol_dir / "model.pt")
        pd.DataFrame(history).to_csv(symbol_dir / "history.csv", index=False)
        results = evaluate_model(test_envs, model, config)
        frame = pd.DataFrame(results)
        frame.to_csv(symbol_dir / "episodes.csv", index=False)
        summary = {**summarize(frame), **training_meta}
        save_json(symbol_dir / "summary.json", summary)
        summaries[symbol] = summary
    save_json(Path(config.output_dir()) / "ppo_summary.json", summaries)
    return summaries


@pyrallis.wrap()
def main(config: TrainConfig) -> None:
    run_train(config)


if __name__ == "__main__":
    main()

