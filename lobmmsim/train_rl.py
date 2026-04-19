from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyrallis
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from .config import RLTrainConfig
from .data import DayData
from .env import MarketMakingEnv
from .pipeline import load_symbol_splits, prepare_run, save_episode_results, summarize_results
from .utils import ensure_dir, save_json
from .baselines import FixedLevelPolicy, OraclePaperPolicy
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


def _inv_softplus(value: float) -> float:
    return float(torch.log(torch.expm1(torch.tensor(value, dtype=torch.float32))).item())


def _init_paper_actor_prior(model: ContinuousActorCritic) -> None:
    with torch.no_grad():
        if model.action_dim != 2:
            return
        # Start near neutral reservation bias, but with a realistically narrow spread so the
        # agent gets enough passive fills to learn from the synthetic market.
        model.alpha_head.weight.zero_()
        model.beta_head.weight.zero_()
        alpha_bias = [_inv_softplus(1.5), _inv_softplus(0.12)]
        if getattr(model, "_action_mode", "absolute") == "residual_fixed1":
            beta_bias = [_inv_softplus(2.0), _inv_softplus(2.0)]
        else:
            beta_bias = [_inv_softplus(1.5), _inv_softplus(4.5)]
        model.alpha_head.bias.copy_(torch.tensor(alpha_bias, dtype=model.alpha_head.bias.dtype))
        model.beta_head.bias.copy_(torch.tensor(beta_bias, dtype=model.beta_head.bias.dtype))
        model.value_head.bias.zero_()


def _teacher_policy(config: RLTrainConfig):
    if config.bc_teacher == "oracle_paper":
        return OraclePaperPolicy(config)
    return FixedLevelPolicy(config, 1)


def _decision_to_action(config: RLTrainConfig, mid: float, inventory: float, ask_price: float, bid_price: float, base_spread: float) -> np.ndarray:
    reservation = 0.5 * (ask_price + bid_price)
    spread = ask_price - bid_price
    if config.action_mode == "signed_absolute":
        bias_action = 0.5 + 0.5 * (reservation - mid) / max(config.max_bias, 1e-8)
    else:
        if inventory == 0:
            delta = 0.0
        else:
            delta = np.sign(inventory) * (mid - reservation)
        bias_action = delta / max(config.max_bias, 1e-8)
    if config.action_mode == "residual_fixed1":
        spread_action = 0.5 + 0.5 * (spread - base_spread) / max(config.residual_spread_range, config.tick_size)
    else:
        spread_action = spread / max(config.max_spread, config.tick_size)
    return np.asarray(
        [
            float(np.clip(bias_action, 0.0, 1.0)),
            float(np.clip(spread_action, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def _collect_imitation_dataset(days: list[DayData], config: RLTrainConfig) -> tuple[TensorDataset | None, dict[str, float]]:
    policy = _teacher_policy(config)
    lob_rows = []
    flat_rows = []
    targets = []
    for day in days:
        env = MarketMakingEnv(day, config)
        for episode_index, span in enumerate(env.selected_episodes(config.max_train_episodes_per_day)):
            env.set_eval_context(episode_index)
            obs = env.reset(span)
            done = False
            while not done:
                event_idx = int(env.episode_decisions[env.step_cursor])
                quote_idx = max(event_idx - env.config.latency, env.config.lookback - 1)
                decision = policy.act(day, quote_idx, env.inventory, env.step_cursor, len(env.episode_decisions))
                lob_rows.append(np.array(obs.lob, copy=True))
                flat_rows.append(np.array(obs.flat, copy=True))
                targets.append(
                    _decision_to_action(
                        config,
                        float(day.midprice[quote_idx]),
                        float(env.inventory),
                        decision.ask_price,
                        decision.bid_price,
                        float(day.spread[quote_idx]),
                    )
                )
                obs, _, done, _ = env.step(
                    {
                        "ask_price": decision.ask_price,
                        "ask_volume": decision.ask_volume,
                        "bid_price": decision.bid_price,
                        "bid_volume": decision.bid_volume,
                        "spread": decision.spread,
                        "reservation": 0.5 * (decision.ask_price + decision.bid_price),
                    }
                )
    if not targets:
        return None, {"bc_samples": 0.0}
    dataset = TensorDataset(
        torch.tensor(np.stack(lob_rows), dtype=torch.float32),
        torch.tensor(np.stack(flat_rows), dtype=torch.float32),
        torch.tensor(np.stack(targets), dtype=torch.float32),
    )
    return dataset, {"bc_samples": float(len(targets))}


def _run_behavior_cloning(model: ContinuousActorCritic, days: list[DayData], config: RLTrainConfig, output_dir: Path) -> dict[str, float]:
    dataset, stats = _collect_imitation_dataset(days, config)
    if dataset is None or config.bc_epochs <= 0:
        return {"bc_samples": 0.0, "bc_final_loss": 0.0}
    loader = DataLoader(dataset, batch_size=config.bc_batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=config.bc_lr)
    history = []
    model.to(config.device)
    model.train()
    for epoch in range(config.bc_epochs):
        losses = []
        for lob, flat, target in loader:
            lob = lob.to(config.device)
            flat = flat.to(config.device)
            target = target.to(config.device)
            dist, _ = model.dist_value(lob, flat)
            loss = F.mse_loss(dist.mean, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch, "bc_loss": float(np.mean(losses) if losses else 0.0)})
    pd.DataFrame(history).to_csv(output_dir / "behavior_cloning_history.csv", index=False)
    return {**stats, "bc_final_loss": float(history[-1]["bc_loss"]) if history else 0.0}


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
        val_days = splits["val"] or splits["test"]
        val_envs = [MarketMakingEnv(day, config) for day in val_days]
        eval_envs = [MarketMakingEnv(day, config) for day in splits["test"]]
        encoder = _build_encoder(config, splits["train"], symbol)
        model = ContinuousActorCritic(encoder, action_dim=2)
        model._action_mode = config.action_mode
        _init_paper_actor_prior(model)
        symbol_dir = ensure_dir(Path(out_dir) / symbol / "ppo")
        bc_summary = _run_behavior_cloning(model, splits["train"], config, symbol_dir)
        pretrain_selected_model = deepcopy(model).cpu()

        def _validation_summary(candidate: ContinuousActorCritic, epoch: int | None = None) -> dict[str, float] | None:
            if not config.ppo_select_best_model or not val_envs:
                return None
            results, _ = evaluate_rl_model(val_envs, candidate, config, output_dir=None, method_name="C_PPO_val")
            frame = pd.DataFrame([result.to_dict() for result in results])
            return summarize_results(frame)

        initial_selection = _validation_summary(pretrain_selected_model)
        initial_metric = float(initial_selection.get(config.ppo_selection_metric, float("-inf"))) if initial_selection else float("-inf")

        model, history, train_runtime = train_ppo(
            train_envs,
            model,
            config,
            select_fn=_validation_summary if config.ppo_select_best_model else None,
        )

        final_selection = _validation_summary(model)
        final_metric = float(final_selection.get(config.ppo_selection_metric, float("-inf"))) if final_selection else float("-inf")
        if initial_selection and initial_metric >= final_metric:
            model.load_state_dict(pretrain_selected_model.state_dict())
            train_runtime["selected_epoch"] = -1.0
            train_runtime["selected_metric"] = initial_metric
            train_runtime["selection_metric_name"] = config.ppo_selection_metric
            train_runtime["selection_source"] = "behavior_cloning"
        elif final_selection:
            train_runtime["selection_source"] = "ppo"

        torch.save(model.state_dict(), symbol_dir / "model.pt")
        pd.DataFrame(history).to_csv(symbol_dir / "history.csv", index=False)
        results, eval_runtime = evaluate_rl_model(eval_envs, model, config, output_dir=symbol_dir, method_name=config.method_name())
        frame = save_episode_results(symbol_dir / "episodes.csv", results)
        summary = summarize_results(frame)
        summary.update(bc_summary)
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
