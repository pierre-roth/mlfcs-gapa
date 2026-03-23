from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd
import pyrallis
import torch

from .config import RLTrainConfig
from .data import DayData
from .env import MarketMakingEnv
from .models import ContinuousActorCritic, SharedStateEncoder, build_backbone
from .pipeline import load_symbol_splits, prepare_run, save_episode_results, summarize_results
from .rl import train_ppo
from .utils import ensure_dir, save_json


def _flat_dim(days: list[DayData], state_mode: str, wo_dynamic_state: bool) -> int:
    sample = days[0]
    if state_mode == "inventory_only":
        return 2
    if state_mode == "handcrafted":
        return sample.handcrafted.shape[1] + 2
    dynamic = 0 if wo_dynamic_state else sample.dynamic.shape[1]
    return dynamic + 2


def _load_matching_state_dict(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    current = module.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    module.load_state_dict(compatible, strict=False)


def _build_encoder(config: RLTrainConfig, days: list[DayData], symbol: str):
    backbone = None
    if not config.wo_lob_state and config.state_mode == "full":
        backbone_name = config.alt_backbone if config.wo_lob_state else config.pretrain_backbone
        backbone = build_backbone(backbone_name, config.lookback)
        ckpt = Path(config.output_dir()) / symbol / "pretrain" / config.backbone_name
        if ckpt.exists():
            state_dict = torch.load(ckpt, map_location="cpu")
            _load_matching_state_dict(backbone, state_dict)
        for param in backbone.parameters():
            param.requires_grad = config.backbone_trainable
    return SharedStateEncoder(backbone, _flat_dim(days, config.state_mode, config.wo_dynamic_state))


def load_trained_ppo(config: RLTrainConfig, symbol: str, days: list[DayData]) -> ContinuousActorCritic:
    encoder = _build_encoder(config, days, symbol)
    model = ContinuousActorCritic(encoder)
    model_path = Path(config.output_dir()) / symbol / "ppo" / config.variant_name() / "model.pt"
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    return model


def run_rl_training(config: RLTrainConfig) -> dict[str, dict[str, float]]:
    config.apply_mode_defaults()
    out_dir = prepare_run(config, label=f"train_{config.algorithm}")
    if config.algorithm == "dqn":
        placeholder = {
            symbol: {
                "status": "skipped",
                "reason": "Discrete dueling DQN intentionally left as a placeholder; the supported replication path is PPO continuous.",
            }
            for symbol in config.symbols
        }
        save_json(out_dir / "train_dqn_placeholder.json", placeholder)
        return placeholder
    summaries: dict[str, dict[str, float]] = {}
    variant_name = config.variant_name()
    for symbol in config.symbols:
        splits = load_symbol_splits(config, symbol)
        train_envs = [MarketMakingEnv(day, config, state_mode=config.state_mode, wo_lob_state=config.wo_lob_state, wo_dynamic_state=config.wo_dynamic_state, reward_mode=config.reward_mode) for day in splits["train"]]
        eval_envs = [MarketMakingEnv(day, config, state_mode=config.state_mode, wo_lob_state=config.wo_lob_state, wo_dynamic_state=config.wo_dynamic_state, reward_mode=config.reward_mode) for day in splits["test"]]
        encoder = _build_encoder(config, splits["train"], symbol)
        symbol_dir = ensure_dir(out_dir / symbol / config.algorithm / variant_name)
        model = ContinuousActorCritic(encoder)
        model, history, train_runtime = train_ppo(train_envs, model, config)
        torch.save(model.state_dict(), symbol_dir / "model.pt")
        trained = model
        pd.DataFrame(history).to_csv(symbol_dir / "history.csv", index=False)
        results, eval_runtime = evaluate_rl_model(eval_envs, trained, config, output_dir=symbol_dir, method_name=config.method_name())
        frame = save_episode_results(symbol_dir / "episodes.csv", results)
        summary = summarize_results(frame)
        summary.update(train_runtime)
        summary.update(eval_runtime)
        save_json(symbol_dir / "summary.json", summary)
        save_json(symbol_dir / "timing.json", {**train_runtime, **eval_runtime, "method": config.method_name()})
        summaries[symbol] = summary
    save_json(out_dir / f"train_{config.algorithm}_{variant_name}.json", summaries)
    return summaries


def evaluate_rl_model(
    envs: list[MarketMakingEnv],
    model: ContinuousActorCritic,
    config: RLTrainConfig,
    output_dir: str | Path | None = None,
    method_name: str | None = None,
):
    model.to(config.device)
    model.eval()
    results = []
    trace_dir = ensure_dir(Path(output_dir) / "traces") if output_dir is not None else None
    inference_steps = 0
    inference_elapsed = 0.0
    for env in envs:
        for episode_index, span in enumerate(env.available_episodes()[: config.max_eval_episodes_per_day]):
            obs = env.reset(span)
            done = False
            attention_rows = []
            while not done:
                flat = torch.tensor(obs.flat[None, :], dtype=torch.float32, device=config.device)
                lob = None if obs.lob is None else torch.tensor(obs.lob[None, :, :], dtype=torch.float32, device=config.device)
                started = perf_counter()
                with torch.no_grad():
                    dist, _ = model.dist_value(lob, flat)
                    action = dist.mean.squeeze(0).cpu().numpy()
                inference_elapsed += perf_counter() - started
                inference_steps += 1
                backbone = getattr(model.encoder, "backbone", None)
                if backbone is not None and getattr(backbone, "last_attention", None) is not None:
                    weights = backbone.last_attention.detach().cpu().numpy()
                    averaged = weights.mean(axis=1).squeeze(1).mean(axis=0)
                    attention_rows.append(averaged)
                obs, _, done, _ = env.step(action)
            method = method_name or config.method_name()
            results.append(env.episode_result(method, episode_index))
            if trace_dir is not None:
                trace = env.episode_trace()
                if not trace.empty:
                    trace.to_csv(trace_dir / f"episode_{episode_index}.csv", index=False)
                if attention_rows:
                    pd.DataFrame(attention_rows).to_csv(trace_dir / f"episode_{episode_index}_attention.csv", index=False)
    runtime = {
        "method": method_name or config.method_name(),
        "inference_steps": float(inference_steps),
        "inference_wall_time_sec": float(inference_elapsed),
        "inference_ms_per_step": float(1000.0 * inference_elapsed / max(inference_steps, 1)),
    }
    if output_dir is not None:
        save_json(Path(output_dir) / "timing.json", runtime)
    return results, runtime


@pyrallis.wrap()
def main(config: RLTrainConfig) -> None:
    run_rl_training(config)


if __name__ == "__main__":
    main()
