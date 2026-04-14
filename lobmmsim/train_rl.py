from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd
import pyrallis
import torch

from .config import RLTrainConfig
from .data import DayData
from .env import MarketMakingEnv
from .pipeline import load_symbol_splits, prepare_run, save_episode_results, summarize_results
from .utils import ensure_dir, save_json
from lobmmx.models import ContinuousActorCritic, SharedStateEncoder, build_backbone
from lobmmx.rl import train_ppo


def _flat_dim(days: list[DayData]) -> int:
    sample = days[0]
    return sample.dynamic.shape[1] + sample.agent_template.shape[1]


def _load_matching_state_dict(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    current = module.state_dict()
    compatible = {key: value for key, value in state_dict.items() if key in current and current[key].shape == value.shape}
    module.load_state_dict(compatible, strict=False)


def _build_encoder(config: RLTrainConfig, days: list[DayData], symbol: str):
    backbone = build_backbone(config.pretrain_backbone, config.lookback)
    ckpt = Path(config.output_dir()) / symbol / "pretrain" / config.backbone_name
    if ckpt.exists():
        _load_matching_state_dict(backbone, torch.load(ckpt, map_location="cpu"))
    for param in backbone.parameters():
        param.requires_grad = config.backbone_trainable
    return SharedStateEncoder(backbone, _flat_dim(days))


def evaluate_rl_model(envs: list[MarketMakingEnv], model: ContinuousActorCritic, config: RLTrainConfig, output_dir: str | Path | None = None, method_name: str = "C_PPO"):
    model.to(config.device)
    model.eval()
    results = []
    trace_dir = ensure_dir(Path(output_dir) / "traces") if output_dir is not None else None
    steps = 0
    elapsed = 0.0
    for env in envs:
        for episode_index, span in enumerate(env.selected_episodes(config.max_eval_episodes_per_day)):
            env.set_eval_context(episode_index)
            obs = env.reset(span)
            done = False
            attention_rows = []
            while not done:
                flat = torch.tensor(obs.flat[None, :], dtype=torch.float32, device=config.device)
                lob = torch.tensor(obs.lob[None, :, :], dtype=torch.float32, device=config.device)
                started = perf_counter()
                with torch.no_grad():
                    dist, _ = model.dist_value(lob, flat)
                    action = dist.mean.squeeze(0).cpu().numpy()
                elapsed += perf_counter() - started
                steps += 1
                backbone = getattr(model.encoder, "backbone", None)
                if backbone is not None and getattr(backbone, "last_attention", None) is not None:
                    weights = backbone.last_attention.detach().cpu().numpy()
                    averaged = weights.mean(axis=1).squeeze(1).mean(axis=0)
                    attention_rows.append(averaged)
                obs, _, done, _ = env.step(action)
            results.append(env.episode_result(method_name, episode_index))
            if trace_dir is not None:
                trace = env.episode_trace()
                if not trace.empty:
                    trace.to_csv(trace_dir / f"episode_{episode_index}.csv", index=False)
                if attention_rows:
                    pd.DataFrame(attention_rows).to_csv(trace_dir / f"episode_{episode_index}_attention.csv", index=False)
    runtime = {
        "method": method_name,
        "inference_steps": float(steps),
        "inference_wall_time_sec": float(elapsed),
        "inference_ms_per_step": float(1000.0 * elapsed / max(steps, 1)),
    }
    if output_dir is not None:
        save_json(Path(output_dir) / "timing.json", runtime)
    return results, runtime


def load_trained_ppo(config: RLTrainConfig, symbol: str, days: list[DayData]) -> ContinuousActorCritic:
    encoder = _build_encoder(config, days, symbol)
    model = ContinuousActorCritic(encoder, action_dim=2)
    state = torch.load(Path(config.output_dir()) / symbol / "ppo" / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    return model


def run_rl_training(config: RLTrainConfig) -> dict[str, dict[str, float]]:
    config.apply_mode_defaults()
    out_dir = prepare_run(config, label="train_ppo")
    summaries: dict[str, dict[str, float]] = {}
    for symbol in config.symbols:
        splits = load_symbol_splits(config, symbol)
        train_envs = [MarketMakingEnv(day, config) for day in splits["train"]]
        eval_envs = [MarketMakingEnv(day, config) for day in splits["test"]]
        encoder = _build_encoder(config, splits["train"], symbol)
        model = ContinuousActorCritic(encoder, action_dim=2)
        symbol_dir = ensure_dir(Path(out_dir) / symbol / "ppo")
        model, history, train_runtime = train_ppo(train_envs, model, config)
        torch.save(model.state_dict(), symbol_dir / "model.pt")
        pd.DataFrame(history).to_csv(symbol_dir / "history.csv", index=False)
        results, eval_runtime = evaluate_rl_model(eval_envs, model, config, output_dir=symbol_dir, method_name=config.method_name())
        frame = save_episode_results(symbol_dir / "episodes.csv", results)
        summary = summarize_results(frame)
        summary.update(train_runtime)
        summary.update(eval_runtime)
        save_json(symbol_dir / "summary.json", summary)
        summaries[symbol] = summary
    save_json(Path(out_dir) / "ppo_summary.json", summaries)
    return summaries


@pyrallis.wrap()
def main(config: RLTrainConfig) -> None:
    run_rl_training(config)


if __name__ == "__main__":
    main()
