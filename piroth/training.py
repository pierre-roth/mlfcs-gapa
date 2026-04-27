from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import DiagnosticsConfig
from .models import AttnLOBEncoder, DuelingDQN, PPOActorCritic, PretrainClassifier, TradingBackbone
from .paper_env import PaperAction, PaperTradingEnv, run_episode
from .paper_features import LOB_COLUMNS, combine_orderbook, lob_tensor_from_values, midprice_direction_labels
from .paper_policies import ContinuousActionPolicy, DiscreteActionPolicy
from .simulator import SyntheticDay


class PretrainDataset(Dataset):
    def __init__(self, days: list[SyntheticDay], config: DiagnosticsConfig) -> None:
        self.days = days
        self.config = config
        self.lob_values = [combine_orderbook(day.ask, day.bid)[LOB_COLUMNS].to_numpy(dtype=np.float32) for day in days]
        self.index: list[tuple[int, int, int]] = []
        for day_idx, day in enumerate(days):
            labels = midprice_direction_labels(day.price["midprice"], config.pretrain_horizon, config.pretrain_threshold)
            day_items = [(int(event_idx), int(label)) for event_idx, label in labels.dropna().astype(int).items() if int(event_idx) >= config.lookback]
            if config.max_pretrain_samples_per_day is not None and len(day_items) > config.max_pretrain_samples_per_day:
                rng = np.random.default_rng(config.seed + day_idx)
                chosen = np.sort(rng.choice(len(day_items), size=config.max_pretrain_samples_per_day, replace=False))
                day_items = [day_items[int(idx)] for idx in chosen]
            for event_idx, label in day_items:
                if event_idx >= config.lookback:
                    self.index.append((day_idx, event_idx, label))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        day_idx, event_idx, label = self.index[item]
        lob = lob_tensor_from_values(self.lob_values[day_idx], event_idx, self.config.lookback)
        return torch.from_numpy(lob), torch.tensor(label, dtype=torch.long)


def train_pretrain_classifier(days: list[SyntheticDay], config: DiagnosticsConfig, output_dir: Path, device: str = "cpu") -> Path:
    _configure_torch(device)
    dataset = PretrainDataset(days, config)
    if len(dataset) == 0:
        raise ValueError("No Attn-LOB pretraining samples were generated; check lookback, horizon, and synthetic event count.")
    loader = DataLoader(
        dataset,
        batch_size=config.torch_batch_size,
        shuffle=True,
        num_workers=_dataloader_workers(device),
        pin_memory=device.startswith("cuda"),
        persistent_workers=_dataloader_workers(device) > 0,
    )
    model = PretrainClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.torch_learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(config.torch_epochs):
        model.train()
        losses = []
        correct = 0
        total = 0
        for lob, label in tqdm(loader, desc=f"pretrain epoch {epoch + 1}", leave=False):
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
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "accuracy": correct / max(total, 1)})
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "attnlob_pretrain.pt"
    torch.save({"model": model.state_dict(), "config": asdict(config), "history": history}, path)
    pd.DataFrame(history).to_csv(output_dir / "attnlob_pretrain_history.csv", index=False)
    return path


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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.torch_learning_rate)
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
                    loss = policy_loss + config.ppo_value_coef * value_loss - config.ppo_entropy_coef * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_value = float(loss.detach().cpu())
        epoch_row = {"epoch": epoch + 1, "reward_mean": float(np.mean([row["reward"] for row in metrics_rows])), "pnl_mean": float(np.mean([row["pnl"] for row in metrics_rows])), "loss": loss_value}
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
    target_model = DuelingDQN(_trading_backbone(config, pretrain_path, device)).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.torch_learning_rate)
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
