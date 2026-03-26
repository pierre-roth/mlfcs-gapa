from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .config import RLTrainConfig
from .env import MarketMakingEnv, Observation
from .models import ContinuousActorCritic, DuelingQNetwork, clone_module


def _obs_to_tensors(obs_batch: list[Observation], device: str) -> tuple[torch.Tensor | None, torch.Tensor]:
    if obs_batch[0].lob is None:
        lob = None
    else:
        lob_np = np.stack([obs.lob for obs in obs_batch]).astype(np.float32, copy=False)
        lob = torch.from_numpy(lob_np).to(device=device, non_blocking=device == "cuda")
    flat_np = np.stack([obs.flat for obs in obs_batch]).astype(np.float32, copy=False)
    flat = torch.from_numpy(flat_np).to(device=device, non_blocking=device == "cuda")
    return lob, flat


def _discount_cumsum(values: list[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    running = 0.0
    for idx in reversed(range(len(values))):
        running = values[idx] + gamma * running
        out[idx] = running
    return out


def _gae(rewards: list[float], values: list[float], dones: list[bool], gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    extended_values = values + [0.0]
    for idx in reversed(range(len(rewards))):
        next_value = 0.0 if dones[idx] else extended_values[idx + 1]
        delta = rewards[idx] + gamma * next_value - extended_values[idx]
        gae = delta + gamma * lam * (0.0 if dones[idx] else gae)
        advantages[idx] = gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def _sample_continuous_action(dist, device: str) -> torch.Tensor:
    if device == "mps":
        cpu_dist = type(dist)(dist.concentration1.detach().cpu(), dist.concentration0.detach().cpu())
        return cpu_dist.sample().to(dist.concentration1.device)
    return dist.sample()


@dataclass
class ReplayItem:
    obs: Observation
    action: int
    reward: float
    next_obs: Observation
    done: bool


@dataclass
class _PendingEpisode:
    env: MarketMakingEnv
    span: tuple[int, int]


@dataclass
class _ActiveEpisode:
    env: MarketMakingEnv
    obs: Observation
    episode_reward: float = 0.0


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[ReplayItem] = deque(maxlen=capacity)

    def append(self, item: ReplayItem) -> None:
        self.buffer.append(item)

    def sample(self, batch_size: int) -> list[ReplayItem]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.buffer)


def train_ppo(
    envs: list[MarketMakingEnv],
    model: ContinuousActorCritic,
    config: RLTrainConfig,
) -> tuple[ContinuousActorCritic, list[dict[str, float]], dict[str, float]]:
    device = config.device
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.ppo_lr)
    history: list[dict[str, float]] = []
    total_steps = 0
    started = perf_counter()
    for epoch in range(config.ppo_epochs):
        rollouts: list[tuple[Observation, np.ndarray, float, float, float, bool]] = []
        rewards_epoch = []
        pending: deque[_PendingEpisode] = deque()
        for env in envs:
            for span in env.selected_episodes(config.max_train_episodes_per_day):
                pending.append(_PendingEpisode(env=env.spawn(), span=span))
        max_active = max(1, config.ppo_rollouts_per_epoch)
        active: list[_ActiveEpisode] = []
        while pending or active:
            while pending and len(active) < max_active:
                episode = pending.popleft()
                active.append(_ActiveEpisode(env=episode.env, obs=episode.env.reset(episode.span)))
            if not active:
                break
            obs_batch = [episode.obs for episode in active]
            with torch.no_grad():
                lob_t, flat_t = _obs_to_tensors(obs_batch, device)
                dist, value = model.dist_value(lob_t, flat_t)
                action = _sample_continuous_action(dist, device)
                log_prob = dist.log_prob(action).sum(dim=-1)
            actions = action.detach().cpu().numpy()
            log_probs = log_prob.detach().cpu().numpy()
            values_mb = value.detach().cpu().numpy()
            next_active: list[_ActiveEpisode] = []
            for idx, episode in enumerate(active):
                obs = episode.obs
                next_obs, reward, done, _ = episode.env.step(actions[idx])
                rollouts.append(
                    (
                        obs,
                        actions[idx],
                        float(log_probs[idx]),
                        float(values_mb[idx]),
                        float(reward),
                        done,
                    )
                )
                episode.episode_reward += float(reward)
                total_steps += 1
                if done:
                    rewards_epoch.append(float(episode.episode_reward))
                else:
                    episode.obs = next_obs
                    next_active.append(episode)
            active = next_active
        if not rollouts:
            break

        model.train()
        rewards = rewards_epoch or [0.0]
        obs_list = [item[0] for item in rollouts]
        actions = np.stack([item[1] for item in rollouts]).astype(np.float32)
        old_log_probs = np.asarray([item[2] for item in rollouts], dtype=np.float32)
        values = [item[3] for item in rollouts]
        reward_stream = [item[4] for item in rollouts]
        dones = [item[5] for item in rollouts]
        advantages, returns = _gae(reward_stream, values, dones, config.gamma, config.gae_lambda)
        if config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)
        actions_t = torch.tensor(actions, dtype=torch.float32)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)
        batch_size = len(obs_list)
        for _ in range(config.ppo_updates):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, config.ppo_minibatch_size):
                batch_idx = indices[start : start + config.ppo_minibatch_size]
                obs_mb = [obs_list[int(idx)] for idx in batch_idx.tolist()]
                lob_mb, flat_mb = _obs_to_tensors(obs_mb, device)
                actions_mb = actions_t[batch_idx].to(device)
                old_log_probs_mb = old_log_probs_t[batch_idx].to(device)
                adv_mb = adv_t[batch_idx].to(device)
                ret_mb = ret_t[batch_idx].to(device)
                dist, value = model.dist_value(lob_mb, flat_mb)
                log_prob = dist.log_prob(actions_mb).sum(dim=-1)
                ratio = torch.exp(log_prob - old_log_probs_mb)
                clipped = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip)
                policy_loss = -torch.min(ratio * adv_mb, clipped * adv_mb).mean()
                value_loss = F.mse_loss(value, ret_mb)
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
        history.append({"epoch": epoch, "reward_mean": float(np.mean(rewards)), "reward_std": float(np.std(rewards))})
    elapsed = perf_counter() - started
    runtime = {
        "train_steps": float(total_steps),
        "train_wall_time_sec": float(elapsed),
        "train_ms_per_step": float(1000.0 * elapsed / max(total_steps, 1)),
    }
    return model, history, runtime


def train_dqn(
    envs: list[MarketMakingEnv],
    model: DuelingQNetwork,
    config: RLTrainConfig,
) -> tuple[DuelingQNetwork, list[dict[str, float]]]:
    device = config.device
    model.to(device)
    target = clone_module(model).to(device)
    target.load_state_dict(model.state_dict())
    optimizer = Adam(model.parameters(), lr=config.dqn_lr)
    replay = ReplayBuffer(config.dqn_replay_size)
    history: list[dict[str, float]] = []
    total_steps = 0
    for epoch in range(config.dqn_epochs):
        epoch_rewards = []
        for env in envs:
            for span in env.selected_episodes(config.max_train_episodes_per_day):
                obs = env.reset(span)
                done = False
                episode_reward = 0.0
                while not done:
                    eps = config.dqn_eps_end + (config.dqn_eps_start - config.dqn_eps_end) * np.exp(-total_steps / max(config.dqn_eps_decay, 1))
                    if np.random.random() < eps:
                        action = np.random.randint(env.num_discrete_actions)
                    else:
                        lob_t, flat_t = _obs_to_tensors([obs], device)
                        with torch.no_grad():
                            action = int(model(lob_t, flat_t).argmax(dim=-1).item())
                    next_obs, reward, done, _ = env.step(action)
                    replay.append(ReplayItem(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done))
                    obs = next_obs
                    total_steps += 1
                    episode_reward += reward
                    if len(replay) >= max(config.dqn_batch_size, config.dqn_warmup_steps):
                        for _ in range(config.dqn_batches_per_epoch):
                            batch = replay.sample(config.dqn_batch_size)
                            obs_lob, obs_flat = _obs_to_tensors([item.obs for item in batch], device)
                            next_lob, next_flat = _obs_to_tensors([item.next_obs for item in batch], device)
                            actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=device)
                            rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device)
                            dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=device)
                            q_values = model(obs_lob, obs_flat).gather(1, actions.unsqueeze(1)).squeeze(1)
                            with torch.no_grad():
                                next_actions = model(next_lob, next_flat).argmax(dim=-1)
                                next_q = target(next_lob, next_flat).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                                target_q = rewards + config.gamma * (1.0 - dones) * next_q
                            loss = F.mse_loss(q_values, target_q)
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                    if total_steps % config.dqn_target_interval == 0:
                        target.load_state_dict(model.state_dict())
                epoch_rewards.append(episode_reward)
        history.append({"epoch": epoch, "reward_mean": float(np.mean(epoch_rewards) if epoch_rewards else 0.0)})
    return model, history
