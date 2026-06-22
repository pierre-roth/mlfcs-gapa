"""Behavioral cloning warm start from an AS teacher."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from mlfcs_gapa.data.schema import LobDataset
from mlfcs_gapa.env.baselines import AvellanedaStoikovStrategy
from mlfcs_gapa.env.gym_env import PaperMarketMakingEnv
from mlfcs_gapa.extensions.as_guidance import (
    as_teacher_action,
    make_as_strategy,
    paper_action_to_env_action,
)
from mlfcs_gapa.paper.constants import PAPER


@dataclass(frozen=True)
class ASDemonstrations:
    observations: dict[str, np.ndarray]
    actions: np.ndarray

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


def collect_as_demonstrations(
    dataset: LobDataset,
    *,
    as_strategy: AvellanedaStoikovStrategy | None = None,
    n_samples: int = 10_000,
    episode_events: int = PAPER.episode_events,
    latency_events: int = 1,
    normalize_actions: bool = True,
    seed: int = 1,
) -> ASDemonstrations:
    """Collect `(observation, AS action)` pairs by rolling out the AS teacher."""

    strategy = as_strategy or make_as_strategy(dataset, episode_events=episode_events)
    env = PaperMarketMakingEnv(
        dataset,
        episode_events=episode_events,
        latency_events=latency_events,
        normalize_actions=normalize_actions,
        random_episode_starts=True,
        seed=seed,
    )

    observations: dict[str, list[np.ndarray]] = {
        "lob_state": [],
        "dynamic_state": [],
        "agent_state": [],
    }
    actions: list[np.ndarray] = []
    episode = 0

    while len(actions) < n_samples:
        obs, _ = env.reset(seed=seed + episode)
        terminated = False
        while not terminated and len(actions) < n_samples:
            progress = (env.current_index - env.episode_start) / max(
                1, env.episode_end - env.episode_start
            )
            teacher = as_teacher_action(
                strategy,
                env.replay,
                env.account,
                env._decision_index(),
                progress,
            )
            env_action = paper_action_to_env_action(
                teacher, normalize_actions=normalize_actions
            )
            for key, value in obs.items():
                observations[key].append(value.copy())
            actions.append(env_action.copy())
            obs, _, terminated, truncated, _ = env.step(env_action)
            terminated = terminated or truncated
        episode += 1

    return ASDemonstrations(
        observations={key: np.stack(values).astype(np.float32) for key, values in observations.items()},
        actions=np.stack(actions).astype(np.float32),
    )


def behavior_clone_ppo_policy(
    model,
    demonstrations: ASDemonstrations,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    mse_weight: float = 1.0,
    nll_weight: float = 0.1,
    entropy_weight: float = 0.0,
    seed: int = 1,
) -> list[dict[str, float | int]]:
    """Warm-start an SB3 PPO policy by imitating AS actions."""

    if demonstrations.size == 0:
        raise ValueError("cannot behavior-clone from an empty demonstration set")

    device = model.policy.device
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    obs_tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in demonstrations.observations.items()
    }
    action_tensor = torch.as_tensor(demonstrations.actions, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    losses: list[dict[str, float | int]] = []

    model.policy.train()
    for epoch in range(epochs):
        order = rng.permutation(demonstrations.size)
        epoch_losses: list[float] = []
        for start in range(0, demonstrations.size, batch_size):
            index = torch.as_tensor(order[start : start + batch_size], device=device)
            batch_obs = {key: value[index] for key, value in obs_tensors.items()}
            batch_actions = action_tensor[index]

            distribution = model.policy.get_distribution(batch_obs)
            log_prob = distribution.log_prob(batch_actions)
            deterministic_actions = distribution.get_actions(deterministic=True)
            entropy = distribution.entropy()
            entropy_term = entropy.mean() if entropy is not None else torch.zeros((), device=device)
            loss = (
                mse_weight * F.mse_loss(deterministic_actions, batch_actions)
                - nll_weight * log_prob.mean()
                - entropy_weight * entropy_term
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        losses.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(epoch_losses)),
                "n_samples": demonstrations.size,
            }
        )
    return losses

