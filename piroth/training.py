from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.utils.data import Dataset
from tqdm import tqdm

from .baselines import calibrate_avellaneda_stoikov
from .config import DiagnosticsConfig
from .models import AttnLOBEncoder, DuelingDQN, PPOActorCritic, PretrainClassifier, TradingBackbone, build_pretrain_classifier
from .paper_env import PaperAction, PaperTradingEnv, run_episode
from .paper_features import LOB_COLUMNS, combine_orderbook, lob_tensor_from_values, midprice_direction_labels
from .paper_policies import AvellanedaStoikovPaperPolicy, ContinuousActionPolicy, DiscreteActionPolicy
from .simulator import SyntheticDay


class PretrainDataset(Dataset):
    def __init__(self, days: list[SyntheticDay], config: DiagnosticsConfig) -> None:
        self.days = days
        self.config = config
        self.lob_values = [combine_orderbook(day.ask, day.bid)[LOB_COLUMNS].to_numpy(dtype=np.float32) for day in days]
        day_indices = []
        event_indices = []
        label_values = []
        for day_idx, day in enumerate(days):
            labels = midprice_direction_labels(day.price["midprice"], config.pretrain_horizon, config.pretrain_threshold)
            if config.pretrain_stable_windows_only:
                labels = labels.loc[_stable_window_mask(day.price["timestamp"], config.stable_windows)]
            valid = labels.dropna().astype(np.int64)
            events = valid.index.to_numpy(dtype=np.int64)
            label_array = valid.to_numpy(dtype=np.int64)
            lookback_mask = events >= config.lookback
            events = events[lookback_mask]
            label_array = label_array[lookback_mask]
            if config.max_pretrain_samples_per_day is not None and len(events) > config.max_pretrain_samples_per_day:
                rng = np.random.default_rng(config.seed + day_idx)
                chosen = np.sort(rng.choice(len(events), size=config.max_pretrain_samples_per_day, replace=False))
                events = events[chosen]
                label_array = label_array[chosen]
            if len(events):
                day_indices.append(np.full(len(events), day_idx, dtype=np.int16))
                event_indices.append(events.astype(np.int32, copy=False))
                label_values.append(label_array.astype(np.int64, copy=False))
        self.day_indices = np.concatenate(day_indices) if day_indices else np.array([], dtype=np.int16)
        self.event_indices = np.concatenate(event_indices) if event_indices else np.array([], dtype=np.int32)
        self.labels = np.concatenate(label_values) if label_values else np.array([], dtype=np.int64)
        self.label_counts = np.bincount(self.labels, minlength=3).astype(int).tolist()

    def __len__(self) -> int:
        return int(self.labels.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        day_idx = int(self.day_indices[item])
        event_idx = int(self.event_indices[item])
        label = int(self.labels[item])
        lob = lob_tensor_from_values(
            self.lob_values[day_idx],
            event_idx,
            self.config.lookback,
            price_z_norm=self.config.lob_price_z_norm,
        )
        return torch.from_numpy(lob), torch.tensor(label, dtype=torch.long)

    def batch(self, items: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        day_indices = self.day_indices[items]
        event_indices = self.event_indices[items]
        labels = self.labels[items]
        lob = np.empty((len(items), self.config.lookback, len(LOB_COLUMNS), 1), dtype=np.float32)
        for day_idx in np.unique(day_indices):
            positions = np.flatnonzero(day_indices == day_idx)
            lob[positions] = _lob_tensor_batch_from_values(
                self.lob_values[int(day_idx)],
                event_indices[positions],
                self.config.lookback,
                price_z_norm=self.config.lob_price_z_norm,
            )
        return torch.from_numpy(lob), torch.from_numpy(labels.astype(np.int64, copy=False))


def train_pretrain_classifier(days: list[SyntheticDay], config: DiagnosticsConfig, output_dir: Path, device: str = "cpu", eval_days: list[SyntheticDay] | None = None) -> Path:
    _configure_torch(device)
    dataset = PretrainDataset(days, config)
    if len(dataset) == 0:
        raise ValueError("No LOB pretraining samples were generated; check lookback, horizon, sampling windows, and event count.")
    eval_dataset = PretrainDataset(eval_days, config) if eval_days else None
    model = build_pretrain_classifier(config.pretrain_model_type, lookback=config.lookback).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.torch_learning_rate)
    class_weights = _pretrain_class_weights(dataset.label_counts, config.pretrain_class_weight_mode, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = []
    for epoch in range(config.torch_epochs):
        model.train()
        losses = []
        correct = 0
        total = 0
        for lob, label in tqdm(
            _iter_pretrain_batches(dataset, config.torch_batch_size, shuffle=True, seed=config.seed + epoch),
            total=_batch_count(len(dataset), config.torch_batch_size),
            desc=f"pretrain epoch {epoch + 1}",
            leave=False,
        ):
            lob = lob.to(device=device, dtype=torch.float32)
            label = label.to(device=device)
            optimizer.zero_grad()
            logits = model(lob)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == label).sum().detach().cpu())
            total += int(label.numel())
        row = {
            "epoch": epoch + 1,
            "model_type": config.pretrain_model_type,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": correct / max(total, 1),
            "train_samples": total,
        }
        if eval_dataset is not None and len(eval_dataset):
            row.update(_evaluate_pretrain_classifier(model, eval_dataset, config.torch_batch_size, criterion, device, prefix="eval"))
        history.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = _pretrain_model_slug(config.pretrain_model_type)
    path = output_dir / ("attnlob_pretrain.pt" if model_slug == "attnlob" else f"{model_slug}_pretrain.pt")
    torch.save({"model": model.state_dict(), "config": asdict(config), "history": history}, path)
    history_path = output_dir / ("attnlob_pretrain_history.csv" if model_slug == "attnlob" else f"{model_slug}_pretrain_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    summary = {
        "model_type": model_slug,
        "checkpoint": str(path),
        "history": str(history_path),
        "train_days": len(days),
        "eval_days": len(eval_days or []),
        "train_label_counts": dataset.label_counts,
        "eval_label_counts": eval_dataset.label_counts if eval_dataset is not None else None,
        "class_weight_mode": config.pretrain_class_weight_mode,
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "final": history[-1],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    with (output_dir / f"{model_slug}_pretrain_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return path


def _evaluate_pretrain_classifier(model: PretrainClassifier, dataset: PretrainDataset, batch_size: int, criterion: nn.Module, device: str, prefix: str) -> dict[str, float | int]:
    model.eval()
    losses = []
    predictions = []
    targets = []
    with torch.no_grad():
        for lob, label in _iter_pretrain_batches(dataset, batch_size, shuffle=False, seed=0):
            lob = lob.to(device=device, dtype=torch.float32)
            label = label.to(device=device)
            logits = model(lob)
            losses.append(float(criterion(logits, label).detach().cpu()))
            predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
            targets.append(label.detach().cpu().numpy())
    pred = np.concatenate(predictions) if predictions else np.array([], dtype=np.int64)
    target = np.concatenate(targets) if targets else np.array([], dtype=np.int64)
    metrics = _classification_metrics(target, pred, num_classes=3)
    return {
        f"{prefix}_loss": float(np.mean(losses)) if losses else float("nan"),
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_precision_macro": metrics["precision_macro"],
        f"{prefix}_recall_macro": metrics["recall_macro"],
        f"{prefix}_f1_macro": metrics["f1_macro"],
        f"{prefix}_samples": int(target.size),
    }


def _batch_count(size: int, batch_size: int) -> int:
    return int((size + max(batch_size, 1) - 1) // max(batch_size, 1))


def _iter_pretrain_batches(dataset: PretrainDataset, batch_size: int, shuffle: bool, seed: int):
    size = len(dataset)
    order = np.arange(size, dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, size, batch_size):
        yield dataset.batch(order[start : start + batch_size])


def _lob_tensor_batch_from_values(values: np.ndarray, event_indices: np.ndarray, lookback: int, *, price_z_norm: bool = False) -> np.ndarray:
    offsets = np.arange(lookback, dtype=np.int32)
    rows = event_indices.astype(np.int32, copy=False)[:, None] - lookback + offsets[None, :]
    windows = values[rows]
    return _normalize_lob_windows(windows, price_z_norm=price_z_norm).reshape(len(event_indices), lookback, len(LOB_COLUMNS), 1)


def _normalize_lob_windows(windows: np.ndarray, *, price_z_norm: bool = False) -> np.ndarray:
    data = windows.astype(np.float32, copy=True)
    mid = (data[:, :, 0].astype(np.float64) + data[:, :, 20].astype(np.float64)) / 2.0
    mid = np.clip(mid, 1e-8, None)
    price_columns: list[int] = []
    for level in range(1, 11):
        ask_base = (level - 1) * 2
        bid_base = 20 + (level - 1) * 2
        data[:, :, ask_base] = data[:, :, ask_base] / mid - 1.0
        data[:, :, bid_base] = data[:, :, bid_base] / mid - 1.0
        price_columns.extend([ask_base, bid_base])
        ask_v = data[:, :, ask_base + 1]
        bid_v = data[:, :, bid_base + 1]
        data[:, :, ask_base + 1] = ask_v / np.maximum(np.max(ask_v, axis=1, keepdims=True), 1.0)
        data[:, :, bid_base + 1] = bid_v / np.maximum(np.max(bid_v, axis=1, keepdims=True), 1.0)
    if price_z_norm:
        for column in price_columns:
            series = data[:, :, column]
            mean = np.mean(series, axis=1, keepdims=True)
            std = np.std(series, axis=1, ddof=1, keepdims=True)
            data[:, :, column] = (series - mean) / (std + 1e-7)
    return data


def _classification_metrics(target: np.ndarray, prediction: np.ndarray, num_classes: int) -> dict[str, float]:
    if target.size == 0:
        return {"accuracy": float("nan"), "precision_macro": float("nan"), "recall_macro": float("nan"), "f1_macro": float("nan")}
    precision = []
    recall = []
    f1 = []
    for klass in range(num_classes):
        tp = float(np.sum((prediction == klass) & (target == klass)))
        fp = float(np.sum((prediction == klass) & (target != klass)))
        fn = float(np.sum((prediction != klass) & (target == klass)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2.0 * p * r / (p + r) if p + r else 0.0)
    return {
        "accuracy": float(np.mean(prediction == target)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }


def _pretrain_model_slug(model_type: str) -> str:
    normalized = model_type.strip().lower().replace("_", "").replace("-", "")
    aliases = {"attn": "attnlob", "fc": "fclob", "conv": "convlob", "deep": "deeplob"}
    return aliases.get(normalized, normalized)


def _pretrain_class_weights(label_counts: list[int], mode: str, device: str) -> torch.Tensor | None:
    normalized = mode.strip().lower().replace("-", "_")
    if normalized in {"", "none", "off", "false"}:
        return None
    counts = torch.tensor(label_counts, dtype=torch.float32, device=device)
    if normalized == "balanced":
        total = counts.sum()
        classes = max(int((counts > 0).sum().item()), 1)
        weights = torch.zeros_like(counts)
        positive = counts > 0
        weights[positive] = total / (classes * counts[positive])
        return weights
    raise ValueError(f"Unknown pretrain_class_weight_mode: {mode!r}")


def _stable_window_mask(timestamps: pd.Series, windows: list[str]) -> np.ndarray:
    clock = timestamps.dt.strftime("%H:%M:%S")
    mask = np.zeros(len(clock), dtype=bool)
    for raw in windows:
        start, end = raw.split("-", maxsplit=1)
        mask |= (clock >= start) & (clock <= end)
    return mask


class PPOModelPolicy:
    name = "C-PPO"

    def __init__(self, model: PPOActorCritic, device: str = "cpu", deterministic: bool = False) -> None:
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.last_log_prob: torch.Tensor | None = None
        self.last_value: torch.Tensor | None = None
        self.last_action_tensor: torch.Tensor | None = None

    def act(self, state, env: PaperTradingEnv) -> PaperAction:
        lob, market, agent = _state_tensors(state, self.device)
        mean, log_std, value = self.model(lob, market, agent)
        dist = Normal(mean, log_std.exp())
        action = mean if self.deterministic else torch.clamp(dist.sample(), -1.0, 1.0)
        self.last_log_prob = dist.log_prob(action).sum(dim=1)
        self.last_value = value
        self.last_action_tensor = action.squeeze(0)
        return ContinuousActionPolicy(action.detach().cpu().numpy()[0]).act(state, env)


class DQNModelPolicy:
    name = "D-DQN"

    def __init__(self, model: DuelingDQN, epsilon: float = 0.0, device: str = "cpu") -> None:
        self.model = model
        self.epsilon = epsilon
        self.device = device

    def act(self, state, env: PaperTradingEnv) -> PaperAction:
        if np.random.random() < self.epsilon:
            action = int(np.random.randint(0, 8))
        else:
            lob, market, agent = _state_tensors(state, self.device)
            with torch.no_grad():
                action = int(torch.argmax(self.model(lob, market, agent), dim=1).cpu().item())
        return DiscreteActionPolicy(action).act(state, env)


def train_ppo(train_days: list[SyntheticDay], config: DiagnosticsConfig, output_dir: Path, pretrain_path: Path | None = None, device: str = "cpu") -> Path:
    _configure_torch(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = PPOActorCritic(
        _trading_backbone(config, pretrain_path, device),
        initial_log_std=config.ppo_initial_log_std,
        initial_spread_bias=config.ppo_initial_spread_bias,
    ).to(device)
    if config.bc_as_init:
        _behavior_clone_ppo_from_as(train_days, config, model, output_dir, device=device)
    optimizer = _optimizer_for_model(model, config)
    history = []
    for epoch in range(config.ppo_epochs):
        episode_specs = [
            (day, episode_index, start, stop)
            for day in train_days
            for episode_index, start, stop in _episode_iter(day, config, config.max_train_episodes_per_day)
        ]
        if config.ppo_shuffle_episodes:
            rng = np.random.default_rng(config.seed + epoch)
            rng.shuffle(episode_specs)
        if config.ppo_rollouts_per_epoch is not None:
            episode_specs = episode_specs[: config.ppo_rollouts_per_epoch]
        metrics_rows = []
        batch_lob: list[torch.Tensor] = []
        batch_market: list[torch.Tensor] = []
        batch_agent: list[torch.Tensor] = []
        batch_action: list[torch.Tensor] = []
        batch_log_prob: list[torch.Tensor] = []
        batch_return: list[torch.Tensor] = []
        batch_advantage: list[torch.Tensor] = []
        policy = PPOModelPolicy(model, device=device)
        for day, episode_index, start, stop in episode_specs:
            env = PaperTradingEnv(day, config, start, stop, episode_index=episode_index, rng_seed=config.seed)
            state = env.reset()
            rewards: list[float] = []
            states = []
            actions: list[torch.Tensor] = []
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            terminal = False
            while not terminal:
                states.append(state)
                action = policy.act(state, env)
                if policy.last_log_prob is not None and policy.last_value is not None and policy.last_action_tensor is not None:
                    actions.append(policy.last_action_tensor.detach().cpu())
                    log_probs.append(policy.last_log_prob.detach().cpu().squeeze(0))
                    values.append(policy.last_value.detach().cpu().squeeze(0))
                result = env.step(action)
                rewards.append(result.reward)
                terminal = result.terminal
                if result.state is not None:
                    state = result.state
            if log_probs and states:
                returns = _discounted_returns(rewards, config.discount, "cpu")
                value_tensor = torch.stack(values)
                advantage = returns - value_tensor
                if advantage.numel() > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
                for state_item, action_tensor, old_log_prob, return_value, advantage_value in zip(states, actions, log_probs, returns, advantage, strict=False):
                    batch_lob.append(torch.from_numpy(state_item.lob_state))
                    batch_market.append(torch.from_numpy(state_item.market_state))
                    batch_agent.append(torch.from_numpy(state_item.agent_state))
                    batch_action.append(action_tensor)
                    batch_log_prob.append(old_log_prob)
                    batch_return.append(return_value)
                    batch_advantage.append(advantage_value)
            metrics_rows.append(asdict(env.metrics()))
        loss_value = 0.0
        if batch_lob:
            dataset_size = len(batch_lob)
            lob_all = torch.stack(batch_lob).to(device=device, dtype=torch.float32)
            market_all = torch.stack(batch_market).to(device=device, dtype=torch.float32)
            agent_all = torch.stack(batch_agent).to(device=device, dtype=torch.float32)
            actions_all = torch.stack(batch_action).to(device=device, dtype=torch.float32)
            old_log_probs_all = torch.stack(batch_log_prob).to(device=device, dtype=torch.float32)
            returns_all = torch.stack(batch_return).to(device=device, dtype=torch.float32)
            advantages_all = torch.stack(batch_advantage).to(device=device, dtype=torch.float32)
            for _ in range(config.ppo_update_epochs):
                order = torch.randperm(dataset_size, device=device)
                for start_idx in range(0, dataset_size, config.torch_batch_size):
                    batch_idx = order[start_idx : start_idx + config.torch_batch_size]
                    lob = lob_all[batch_idx]
                    market = market_all[batch_idx]
                    agent = agent_all[batch_idx]
                    actions_t = actions_all[batch_idx]
                    old_log_probs = old_log_probs_all[batch_idx]
                    returns_t = returns_all[batch_idx]
                    advantages = advantages_all[batch_idx]
                    mean, log_std, values_t = model(lob, market, agent)
                    dist = Normal(mean, log_std.exp())
                    log_probs_t = dist.log_prob(actions_t).sum(dim=1)
                    entropy = dist.entropy().sum(dim=1).mean()
                    ratio = torch.exp(log_probs_t - old_log_probs)
                    unclipped = ratio * advantages
                    clipped = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = F.mse_loss(values_t, returns_t)
                    entropy_coef = _linear_schedule(
                        config.ppo_entropy_coef,
                        config.ppo_entropy_coef_final,
                        epoch,
                        config.ppo_epochs,
                    )
                    loss = policy_loss + config.ppo_value_coef * value_loss - entropy_coef * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_value = float(loss.detach().cpu())
        epoch_row = {
            "epoch": epoch + 1,
            "reward_mean": float(np.mean([row["reward"] for row in metrics_rows])),
            "pnl_mean": float(np.mean([row["pnl"] for row in metrics_rows])),
            "loss": loss_value,
            "entropy_coef": _linear_schedule(config.ppo_entropy_coef, config.ppo_entropy_coef_final, epoch, config.ppo_epochs),
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(output_dir / "c_ppo_history.csv", index=False)
        print(
            f"PPO epoch {epoch_row['epoch']}: reward_mean={epoch_row['reward_mean']:.4f} "
            f"pnl_mean={epoch_row['pnl_mean']:.4f} loss={epoch_row['loss']:.6f}",
            flush=True,
        )
    path = output_dir / "c_ppo.pt"
    torch.save({"model": model.state_dict(), "config": asdict(config), "history": history}, path)
    pd.DataFrame(history).to_csv(output_dir / "c_ppo_history.csv", index=False)
    return path


def train_dqn(train_days: list[SyntheticDay], config: DiagnosticsConfig, output_dir: Path, pretrain_path: Path | None = None, device: str = "cpu") -> Path:
    _configure_torch(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = DuelingDQN(_trading_backbone(config, pretrain_path, device)).to(device)
    if config.bc_as_init:
        _behavior_clone_dqn_from_as(train_days, config, model, output_dir, device=device)
    target_model = DuelingDQN(_trading_backbone(config, pretrain_path, device)).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = _optimizer_for_model(model, config)
    criterion = nn.SmoothL1Loss()
    replay: deque[tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, bool]] = deque(maxlen=config.dqn_replay_size)
    history = []
    epsilon = config.dqn_epsilon_start
    update_steps = 0
    collection_steps = 0
    for epoch in range(config.torch_epochs):
        losses = []
        rows = []
        for day in train_days:
            for episode_index, start, stop in _episode_iter(day, config, config.max_train_episodes_per_day):
                env = PaperTradingEnv(day, config, start, stop, episode_index=episode_index, rng_seed=config.seed)
                state = env.reset()
                terminal = False
                while not terminal:
                    lob, market, agent = _state_tensors(state, device)
                    with torch.no_grad():
                        q_values = model(lob, market, agent)
                    action_idx = int(torch.argmax(q_values, dim=1).cpu().item()) if np.random.random() >= epsilon else int(np.random.randint(0, 8))
                    result = env.step(DiscreteActionPolicy(action_idx).act(state, env))
                    next_state = result.state if result.state is not None else state
                    collection_steps += 1
                    replay.append(
                        (
                            state.lob_state,
                            state.market_state,
                            state.agent_state,
                            action_idx,
                            float(result.reward),
                            next_state.lob_state,
                            next_state.market_state,
                            next_state.agent_state,
                            result.terminal,
                        )
                    )
                    if len(replay) >= config.dqn_min_replay and collection_steps % config.dqn_update_interval == 0:
                        sample_idx = np.random.choice(len(replay), size=min(config.torch_batch_size, len(replay)), replace=False)
                        batch = [replay[int(i)] for i in sample_idx]
                        lob_b = torch.from_numpy(np.stack([item[0] for item in batch])).to(device=device, dtype=torch.float32)
                        market_b = torch.from_numpy(np.stack([item[1] for item in batch])).to(device=device, dtype=torch.float32)
                        agent_b = torch.from_numpy(np.stack([item[2] for item in batch])).to(device=device, dtype=torch.float32)
                        action_b = torch.tensor([item[3] for item in batch], dtype=torch.long, device=device)
                        reward_b = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=device)
                        next_lob_b = torch.from_numpy(np.stack([item[5] for item in batch])).to(device=device, dtype=torch.float32)
                        next_market_b = torch.from_numpy(np.stack([item[6] for item in batch])).to(device=device, dtype=torch.float32)
                        next_agent_b = torch.from_numpy(np.stack([item[7] for item in batch])).to(device=device, dtype=torch.float32)
                        terminal_b = torch.tensor([item[8] for item in batch], dtype=torch.float32, device=device)
                        prediction = model(lob_b, market_b, agent_b).gather(1, action_b[:, None]).squeeze(1)
                        with torch.no_grad():
                            target = reward_b + config.discount * (1.0 - terminal_b) * target_model(next_lob_b, next_market_b, next_agent_b).max(dim=1).values
                        loss = criterion(prediction, target)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        update_steps += 1
                        if update_steps % config.dqn_target_update_steps == 0:
                            target_model.load_state_dict(model.state_dict())
                        losses.append(float(loss.detach().cpu()))
                    terminal = result.terminal
                    if result.state is not None:
                        state = result.state
                rows.append(asdict(env.metrics()))
        epsilon = max(config.dqn_epsilon_end, epsilon * config.dqn_epsilon_decay)
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "epsilon": epsilon,
            "reward_mean": float(np.mean([item["reward"] for item in rows])) if rows else 0.0,
            "pnl_mean": float(np.mean([item["pnl"] for item in rows])) if rows else 0.0,
            "updates": update_steps,
            "collection_steps": collection_steps,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "d_dqn_history.csv", index=False)
        print(
            "DQN epoch "
            f"{row['epoch']}: reward_mean={row['reward_mean']:.4f} "
            f"pnl_mean={row['pnl_mean']:.4f} loss={row['loss']:.6f} "
            f"epsilon={row['epsilon']:.4f} updates={row['updates']}",
            flush=True,
        )
    torch.save({"model": model.state_dict(), "config": asdict(config), "history": history}, output_dir / "d_dqn.pt")
    pd.DataFrame(history).to_csv(output_dir / "d_dqn_history.csv", index=False)
    return output_dir / "d_dqn.pt"


def evaluate_trained_policy(days: list[SyntheticDay], config: DiagnosticsConfig, checkpoint: Path, kind: str, output_dir: Path, device: str = "cpu") -> pd.DataFrame:
    _configure_torch(device)
    if kind == "ppo":
        model = PPOActorCritic(
            _trading_backbone(config, None, device),
            initial_log_std=config.ppo_initial_log_std,
            initial_spread_bias=config.ppo_initial_spread_bias,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
        policy = PPOModelPolicy(model, device=device, deterministic=True)
    elif kind == "dqn":
        model = DuelingDQN(_trading_backbone(config, None, device)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
        policy = DQNModelPolicy(model, epsilon=0.0, device=device)
    else:
        raise ValueError(f"Unknown policy kind: {kind}")
    rows = []
    for day in days:
        for episode_index, start, stop in _episode_iter(day, config, config.max_eval_episodes_per_day):
            env = PaperTradingEnv(day, config, start, stop, episode_index=episode_index, rng_seed=config.seed)
            rows.append(asdict(run_episode(env, policy)))
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"{kind}_episodes.csv", index=False)
    return frame


def _behavior_clone_ppo_from_as(train_days: list[SyntheticDay], config: DiagnosticsConfig, model: PPOActorCritic, output_dir: Path, *, device: str) -> None:
    samples = _as_supervision_samples(train_days, config, for_discrete=False)
    if not samples:
        return
    _set_bc_trainable(model, config)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.torch_learning_rate)
    history = []
    lob_all, market_all, agent_all, target_all = _stack_bc_samples(samples, device, continuous=True)
    dataset_size = target_all.shape[0]
    for epoch in range(config.bc_as_epochs):
        losses = []
        order = torch.randperm(dataset_size, device=device)
        for start_idx in range(0, dataset_size, config.torch_batch_size):
            batch_idx = order[start_idx : start_idx + config.torch_batch_size]
            mean, _, _ = model(lob_all[batch_idx], market_all[batch_idx], agent_all[batch_idx])
            loss = F.mse_loss(mean, target_all[batch_idx]) * config.bc_as_loss_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "samples": int(dataset_size)})
    _set_model_trainable(model, True)
    pd.DataFrame(history).to_csv(output_dir / "as_bc_ppo_history.csv", index=False)


def _behavior_clone_dqn_from_as(train_days: list[SyntheticDay], config: DiagnosticsConfig, model: DuelingDQN, output_dir: Path, *, device: str) -> None:
    samples = _as_supervision_samples(train_days, config, for_discrete=True)
    if not samples:
        return
    _set_bc_trainable(model, config)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.torch_learning_rate)
    history = []
    lob_all, market_all, agent_all, target_all = _stack_bc_samples(samples, device, continuous=False)
    dataset_size = target_all.shape[0]
    for epoch in range(config.bc_as_epochs):
        losses = []
        correct = 0
        order = torch.randperm(dataset_size, device=device)
        for start_idx in range(0, dataset_size, config.torch_batch_size):
            batch_idx = order[start_idx : start_idx + config.torch_batch_size]
            logits = model(lob_all[batch_idx], market_all[batch_idx], agent_all[batch_idx])
            target = target_all[batch_idx].long()
            loss = F.cross_entropy(logits, target) * config.bc_as_loss_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == target).sum().detach().cpu())
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "accuracy": correct / max(dataset_size, 1), "samples": int(dataset_size)})
    _set_model_trainable(model, True)
    pd.DataFrame(history).to_csv(output_dir / "as_bc_dqn_history.csv", index=False)


def _as_supervision_samples(train_days: list[SyntheticDay], config: DiagnosticsConfig, *, for_discrete: bool) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | int]]:
    calibration = calibrate_avellaneda_stoikov(train_days, config)
    teacher = AvellanedaStoikovPaperPolicy(calibration)
    rng = np.random.default_rng(config.seed)
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | int]] = []
    for day_idx, day in enumerate(train_days):
        day_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | int]] = []
        for episode_index, start, stop in _episode_iter(day, config, config.max_train_episodes_per_day):
            env = PaperTradingEnv(day, config, start, stop, episode_index=episode_index, rng_seed=config.seed)
            state = env.reset()
            terminal = False
            while not terminal:
                action = teacher.act(state, env)
                target: np.ndarray | int
                if for_discrete:
                    target = _nearest_discrete_action(state, env, action)
                else:
                    target = _continuous_action_target(state, env, action)
                day_samples.append((state.lob_state, state.market_state, state.agent_state, target))
                result = env.step(action)
                terminal = result.terminal
                if result.state is not None:
                    state = result.state
        limit = config.bc_as_max_samples_per_day
        if limit is not None and len(day_samples) > limit:
            chosen = np.sort(rng.choice(len(day_samples), size=limit, replace=False))
            day_samples = [day_samples[int(idx)] for idx in chosen]
        samples.extend(day_samples)
    return samples


def _stack_bc_samples(
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | int]],
    device: str,
    *,
    continuous: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lob = torch.from_numpy(np.stack([item[0] for item in samples])).to(device=device, dtype=torch.float32)
    market = torch.from_numpy(np.stack([item[1] for item in samples])).to(device=device, dtype=torch.float32)
    agent = torch.from_numpy(np.stack([item[2] for item in samples])).to(device=device, dtype=torch.float32)
    if continuous:
        target = torch.from_numpy(np.stack([item[3] for item in samples]).astype(np.float32)).to(device=device, dtype=torch.float32)
    else:
        target = torch.tensor([int(item[3]) for item in samples], dtype=torch.long, device=device)
    return lob, market, agent, target


def _continuous_action_target(state, env: PaperTradingEnv, action: PaperAction) -> np.ndarray:
    quote_idx = max(env.event_idx - env.config.latency, 0)
    mid = float(env.day.price.iloc[quote_idx]["midprice"])
    spread = max(action.ask_price - action.bid_price, env.config.symbol_spec.tick_size)
    reservation = 0.5 * (action.ask_price + action.bid_price)
    max_bias = max(env.config.max_bias, env.config.symbol_spec.tick_size)
    max_spread = max(env.config.max_spread, env.config.symbol_spec.tick_size)
    if env.config.continuous_action_mode == "author":
        raw_spread = 2.0 * (spread / max_spread) - 1.0
        raw_bias = 2.0 * (abs(reservation - mid) / max_bias) - 1.0
    elif env.config.continuous_action_mode == "author_raw":
        raw_spread = spread / max_spread
        raw_bias = (reservation - mid) / max_bias
    elif env.config.continuous_action_mode == "bounded":
        raw_spread = 2.0 * ((spread - env.config.symbol_spec.tick_size) / max(max_spread - env.config.symbol_spec.tick_size, 1e-8)) - 1.0
        if env.inventory > 0:
            raw_bias = -abs(reservation - mid) / max_bias
        elif env.inventory < 0:
            raw_bias = abs(reservation - mid) / max_bias
        else:
            raw_bias = (reservation - mid) / max_bias
    else:
        raise ValueError(f"Unknown continuous_action_mode: {env.config.continuous_action_mode}")
    return np.asarray([np.clip(raw_bias, -1.0, 1.0), np.clip(raw_spread, -1.0, 1.0)], dtype=np.float32)


def _nearest_discrete_action(state, env: PaperTradingEnv, teacher_action: PaperAction) -> int:
    best_action = 0
    best_score = float("inf")
    tick = env.config.symbol_spec.tick_size
    for action_idx in range(8):
        candidate = DiscreteActionPolicy(action_idx).act(state, env)
        score = 0.0
        if teacher_action.ask_volume and candidate.ask_volume:
            score += abs(candidate.ask_price - teacher_action.ask_price) / tick
        elif teacher_action.ask_volume != candidate.ask_volume:
            score += 100.0
        if teacher_action.bid_volume and candidate.bid_volume:
            score += abs(candidate.bid_price - teacher_action.bid_price) / tick
        elif teacher_action.bid_volume != candidate.bid_volume:
            score += 100.0
        if score < best_score:
            best_action = action_idx
            best_score = score
    return best_action


def _set_bc_trainable(model: nn.Module, config: DiagnosticsConfig) -> None:
    _set_model_trainable(model, True)
    if not config.bc_as_freeze_backbone:
        return
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    if config.bc_as_freeze_encoder_only:
        for parameter in backbone.encoder.parameters():
            parameter.requires_grad = False
        return
    for parameter in backbone.parameters():
        parameter.requires_grad = False


def _set_model_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    for parameter in backbone.parameters():
        parameter.requires_grad = trainable


def _optimizer_for_model(model: nn.Module, config: DiagnosticsConfig) -> torch.optim.Optimizer:
    base_lr = float(config.torch_learning_rate)
    encoder_lr = base_lr * float(config.torch_encoder_learning_rate_scale)
    backbone_lr = base_lr * float(config.torch_backbone_learning_rate_scale)
    groups: dict[str, dict[str, object]] = {
        "encoder": {"params": [], "lr": encoder_lr},
        "backbone": {"params": [], "lr": backbone_lr},
        "head": {"params": [], "lr": base_lr},
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.encoder."):
            groups["encoder"]["params"].append(parameter)  # type: ignore[union-attr]
        elif name.startswith("backbone."):
            groups["backbone"]["params"].append(parameter)  # type: ignore[union-attr]
        else:
            groups["head"]["params"].append(parameter)  # type: ignore[union-attr]
    param_groups = [
        {"params": group["params"], "lr": group["lr"]}
        for group in groups.values()
        if group["params"]
    ]
    return torch.optim.Adam(param_groups, lr=base_lr)


def _linear_schedule(start: float, final: float | None, step: int, total_steps: int) -> float:
    if final is None or total_steps <= 1:
        return float(start)
    fraction = step / max(total_steps - 1, 1)
    return float(start + fraction * (final - start))


def _state_tensors(state, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lob = torch.from_numpy(state.lob_state[None]).to(device=device, dtype=torch.float32)
    market = torch.from_numpy(state.market_state[None]).to(device=device, dtype=torch.float32)
    agent = torch.from_numpy(state.agent_state[None]).to(device=device, dtype=torch.float32)
    return lob, market, agent


def _configure_torch(device: str) -> None:
    if device.startswith("cuda") and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _dataloader_workers(device: str) -> int:
    return 4 if device.startswith("cuda") else 2


def _loaded_encoder(path: Path | None, device: str) -> AttnLOBEncoder:
    encoder = AttnLOBEncoder()
    if path is not None and path.exists():
        payload = torch.load(path, map_location=device)
        model = PretrainClassifier()
        model.load_state_dict(payload["model"])
        encoder.load_state_dict(model.encoder.state_dict())
    return encoder


def _trading_backbone(config: DiagnosticsConfig, pretrain_path: Path | None, device: str) -> TradingBackbone:
    return TradingBackbone(
        encoder=_loaded_encoder(pretrain_path, device),
        include_lob=config.include_lob_state,
        include_market=config.include_market_state,
        include_agent=config.include_agent_state,
        alias_market_to_agent=config.author_market_state_alias,
    )


def _discounted_returns(rewards: list[float], discount: float, device: str) -> torch.Tensor:
    values = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + discount * running
        values.append(running)
    values.reverse()
    returns = torch.tensor(values, dtype=torch.float32, device=device)
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    return returns


def _episode_iter(day: SyntheticDay, config: DiagnosticsConfig, max_episodes: int | None):
    clock = day.price["timestamp"].dt.strftime("%H:%M:%S")
    mask = np.zeros(len(clock), dtype=bool)
    for raw in config.stable_windows:
        start_s, end_s = raw.split("-", maxsplit=1)
        mask |= (clock >= start_s) & (clock <= end_s)
    idx = np.flatnonzero(mask)
    episode_index = 0
    emitted_for_day = 0
    for offset in range(0, len(idx), config.episode_length):
        window = idx[offset : offset + config.episode_length]
        if len(window) == config.episode_length:
            yield episode_index, int(window[0]), int(window[-1]) + 1
            episode_index += 1
            emitted_for_day += 1
            if max_episodes is not None and emitted_for_day >= max_episodes:
                break
