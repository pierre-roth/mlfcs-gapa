from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch.optim import Adam

from .config import TrainConfig


@dataclass
class RolloutBatch:
    lob: torch.Tensor
    flat: torch.Tensor
    actions: torch.Tensor
    old_logprob: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def _discounted_gae(rewards: list[float], values: list[float], dones: list[bool], gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        mask = 0.0 if dones[idx] else 1.0
        delta = rewards[idx] + gamma * next_value * mask - values[idx]
        gae = delta + gamma * lam * mask * gae
        advantages[idx] = gae
        next_value = values[idx]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return returns, advantages


def _batch_from_rollouts(rollouts: list[dict], config: TrainConfig) -> RolloutBatch:
    rewards = [step["reward"] for step in rollouts]
    values = [step["value"] for step in rollouts]
    dones = [step["done"] for step in rollouts]
    returns, advantages = _discounted_gae(rewards, values, dones, config.gamma, config.gae_lambda)
    if config.normalize_advantages and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-8)
    return RolloutBatch(
        lob=torch.tensor(np.stack([step["lob"] for step in rollouts]), dtype=torch.float32, device=config.device),
        flat=torch.tensor(np.stack([step["flat"] for step in rollouts]), dtype=torch.float32, device=config.device),
        actions=torch.tensor(np.stack([step["action"] for step in rollouts]), dtype=torch.float32, device=config.device),
        old_logprob=torch.tensor([step["logprob"] for step in rollouts], dtype=torch.float32, device=config.device),
        returns=torch.tensor(returns, dtype=torch.float32, device=config.device),
        advantages=torch.tensor(advantages, dtype=torch.float32, device=config.device),
    )


def train_ppo(envs, model, config: TrainConfig, select_fn=None):
    model.to(config.device)
    optimizer = Adam(model.parameters(), lr=config.ppo_lr)
    history = []
    episode_queue = deque()
    for env in envs:
        for span in env.selected_episodes(config.max_train_episodes_per_day):
            episode_queue.append((env, span))
    if not episode_queue:
        raise RuntimeError("No training episodes available")
    best_metric = float("-inf")
    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    best_epoch = -1
    for epoch in range(config.ppo_epochs):
        rollout_steps = []
        episode_rewards = []
        for ep_idx in range(config.ppo_rollouts_per_epoch):
            env, span = episode_queue[0]
            episode_queue.rotate(-1)
            obs = env.reset(span)
            done = False
            ep_reward = 0.0
            ep_steps = 0
            while not done:
                lob = torch.tensor(obs.lob[None, :, :], dtype=torch.float32, device=config.device)
                flat = torch.tensor(obs.flat[None, :], dtype=torch.float32, device=config.device)
                with torch.no_grad():
                    dist, value = model.dist_value(lob, flat)
                    action = dist.sample().squeeze(0)
                    logprob = dist.log_prob(action).sum(dim=-1).item()
                next_obs, reward, done, _ = env.step(action.cpu().numpy())
                rollout_steps.append(
                    {
                        "lob": obs.lob,
                        "flat": obs.flat,
                        "action": action.cpu().numpy(),
                        "logprob": logprob,
                        "value": float(value.item()),
                        "reward": float(reward),
                        "done": bool(done),
                    }
                )
                ep_reward += float(reward)
                ep_steps += 1
                obs = next_obs
            episode_rewards.append(ep_reward)
        batch = _batch_from_rollouts(rollout_steps, config)
        losses = []
        for _ in range(config.ppo_updates):
            perm = torch.randperm(batch.actions.shape[0], device=config.device)
            for start in range(0, batch.actions.shape[0], config.ppo_minibatch_size):
                idx = perm[start : start + config.ppo_minibatch_size]
                dist, value = model.dist_value(batch.lob[idx], batch.flat[idx])
                logprob = dist.log_prob(batch.actions[idx]).sum(dim=-1)
                ratio = torch.exp(logprob - batch.old_logprob[idx])
                unclipped = ratio * batch.advantages[idx]
                clipped = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * batch.advantages[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * torch.square(value - batch.returns[idx]).mean()
                entropy_bonus = dist.entropy().sum(dim=-1).mean()
                loss = policy_loss + value_loss - 1e-3 * entropy_bonus
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                losses.append(float(loss.item()))
        metric = None
        if select_fn is not None:
            summary = select_fn(model, epoch)
            if summary is not None:
                metric = float(summary.get(config.ppo_selection_metric, float("-inf")))
                if metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        mean_loss = float(np.mean(losses) if losses else 0.0)
        metric_str = f"{metric:.4f}" if metric is not None else "N/A"
        ep_arr = np.array(episode_rewards, dtype=np.float32)
        ep_mean = float(ep_arr.mean()) if len(ep_arr) > 0 else 0.0
        ep_std = float(ep_arr.std()) if len(ep_arr) > 0 else 0.0
        ep_best = float(ep_arr.max()) if len(ep_arr) > 0 else 0.0
        trades_str = "N/A"
        fill_str = "N/A"
        if select_fn is not None and summary is not None:
            trades_val = summary.get("trades_mean")
            fill_val = summary.get("fill_rate_mean")
            if trades_val is not None:
                trades_str = f"{trades_val:.2f}"
            if fill_val is not None:
                fill_str = f"{fill_val:.3f}"
        print(
            f"  [ppo] epoch {epoch+1}/{config.ppo_epochs} "
            f"loss={mean_loss:.4f} "
            f"ep_reward={ep_mean:.2f}±{ep_std:.2f} best={ep_best:.2f} "
            f"val_{config.ppo_selection_metric}={metric_str} "
            f"val_trades={trades_str} val_fill={fill_str} "
            f"best_epoch={best_epoch}"
        )
        history.append({"epoch": epoch, "loss": mean_loss, "selected_metric": metric})
    model.load_state_dict(best_state)
    return model, history, {"selected_epoch": float(best_epoch), "selected_metric": float(best_metric) if best_metric > float("-inf") else None}

