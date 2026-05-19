"""Dueling Double DQN trainer for the paper's discrete-action agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from mlfcs_gapa.env.discrete_env import PaperDiscreteMarketMakingEnv
from mlfcs_gapa.models.attn_lob import AttnLOBEncoder
from mlfcs_gapa.paper.constants import PAPER
from mlfcs_gapa.training.ppo import load_attn_lob_encoder


Observation = dict[str, np.ndarray]
TensorObservation = dict[str, torch.Tensor]


@dataclass(frozen=True)
class DuelingDQNConfig:
    total_timesteps: int = 1_000
    learning_starts: int = 100
    buffer_size: int = 10_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99
    train_frequency: int = 1
    target_update_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_fraction: float = 0.5
    seed: int = 1
    features_dim: int = 128


@dataclass(frozen=True)
class DuelingDQNTrainResult:
    losses: list[float]
    final_epsilon: float
    updates: int


class DuelingDQN(nn.Module):
    """Attn-LOB dueling Q-network: Q(s,a)=V(s)+A(s,a)-mean_a A(s,a)."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        n_actions: int,
        *,
        features_dim: int = 128,
        encoder_checkpoint: str | None = None,
        freeze_encoder: bool = False,
        lob_mode: str = "attn",
        use_dynamic_state: bool = True,
        use_agent_state: bool = True,
    ) -> None:
        super().__init__()
        self.lob_mode = _validate_lob_mode(lob_mode)
        self.use_dynamic_state = use_dynamic_state
        self.use_agent_state = use_agent_state

        lob_dim = 0
        if self.lob_mode == "attn":
            self.lob_encoder: nn.Module = AttnLOBEncoder()
            if encoder_checkpoint:
                load_attn_lob_encoder(self.lob_encoder, Path(encoder_checkpoint))
            if freeze_encoder:
                for parameter in self.lob_encoder.parameters():
                    parameter.requires_grad = False
            lob_dim = 64
        elif self.lob_mode == "mlp":
            if encoder_checkpoint:
                raise ValueError("encoder_checkpoint is only valid with lob_mode='attn'")
            self.lob_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(PAPER.window_length * PAPER.lob_width, 64),
                nn.LeakyReLU(0.01),
            )
            lob_dim = 64
        else:
            if encoder_checkpoint:
                raise ValueError("encoder_checkpoint is only valid with lob_mode='attn'")
            self.lob_encoder = nn.Identity()

        dynamic_dim = int(observation_space["dynamic_state"].shape[0]) if use_dynamic_state else 0
        agent_dim = int(observation_space["agent_state"].shape[0]) if use_agent_state else 0
        if lob_dim + dynamic_dim + agent_dim == 0:
            raise ValueError("at least one observation component must be enabled")
        self.trunk = nn.Sequential(
            nn.Linear(lob_dim + dynamic_dim + agent_dim, features_dim),
            nn.LeakyReLU(0.01),
        )
        self.value = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(features_dim, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(features_dim, n_actions),
        )

    def forward(self, observations: TensorObservation) -> torch.Tensor:
        observation_features = []
        if self.lob_mode != "none":
            observation_features.append(self.lob_encoder(observations["lob_state"].float()))
        if self.use_dynamic_state:
            observation_features.append(observations["dynamic_state"].float())
        if self.use_agent_state:
            observation_features.append(observations["agent_state"].float())
        features = self.trunk(torch.cat(observation_features, dim=1))
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.position = 0
        self.length = 0
        self.lob = np.zeros((size, *PAPER.lob_window_shape), dtype=np.float32)
        self.dynamic = np.zeros((size, 24), dtype=np.float32)
        self.agent = np.zeros((size, 2), dtype=np.float32)
        self.next_lob = np.zeros((size, *PAPER.lob_window_shape), dtype=np.float32)
        self.next_dynamic = np.zeros((size, 24), dtype=np.float32)
        self.next_agent = np.zeros((size, 2), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

    def add(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        idx = self.position
        self.lob[idx] = observation["lob_state"]
        self.dynamic[idx] = observation["dynamic_state"]
        self.agent[idx] = observation["agent_state"]
        self.next_lob[idx] = next_observation["lob_state"]
        self.next_dynamic[idx] = next_observation["dynamic_state"]
        self.next_agent[idx] = next_observation["agent_state"]
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.position = (self.position + 1) % self.size
        self.length = min(self.length + 1, self.size)

    def sample(self, batch_size: int, device: torch.device) -> tuple[TensorObservation, ...]:
        indices = self.rng.integers(0, self.length, size=batch_size)
        observations = {
            "lob_state": torch.from_numpy(self.lob[indices]).to(device),
            "dynamic_state": torch.from_numpy(self.dynamic[indices]).to(device),
            "agent_state": torch.from_numpy(self.agent[indices]).to(device),
        }
        next_observations = {
            "lob_state": torch.from_numpy(self.next_lob[indices]).to(device),
            "dynamic_state": torch.from_numpy(self.next_dynamic[indices]).to(device),
            "agent_state": torch.from_numpy(self.next_agent[indices]).to(device),
        }
        actions = torch.from_numpy(self.actions[indices]).to(device)
        rewards = torch.from_numpy(self.rewards[indices]).to(device)
        dones = torch.from_numpy(self.dones[indices]).to(device)
        return observations, next_observations, actions, rewards, dones


def train_dueling_dqn(
    env: PaperDiscreteMarketMakingEnv,
    *,
    config: DuelingDQNConfig = DuelingDQNConfig(),
    encoder_checkpoint: str | None = None,
    freeze_encoder: bool = False,
    lob_mode: str = "attn",
    use_dynamic_state: bool = True,
    use_agent_state: bool = True,
    device: str = "cpu",
) -> tuple[DuelingDQN, DuelingDQNTrainResult]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    torch_device = torch.device(device)
    policy = DuelingDQN(
        env.observation_space,
        int(env.action_space.n),
        features_dim=config.features_dim,
        encoder_checkpoint=encoder_checkpoint,
        freeze_encoder=freeze_encoder,
        lob_mode=lob_mode,
        use_dynamic_state=use_dynamic_state,
        use_agent_state=use_agent_state,
    ).to(torch_device)
    target = DuelingDQN(
        env.observation_space,
        int(env.action_space.n),
        features_dim=config.features_dim,
        encoder_checkpoint=encoder_checkpoint,
        freeze_encoder=freeze_encoder,
        lob_mode=lob_mode,
        use_dynamic_state=use_dynamic_state,
        use_agent_state=use_agent_state,
    ).to(torch_device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    replay = ReplayBuffer(config.buffer_size, config.seed)
    observation, _ = env.reset(seed=config.seed)
    losses: list[float] = []
    updates = 0

    for step in range(config.total_timesteps):
        epsilon = _epsilon_at_step(config, step)
        if rng.random() < epsilon:
            action = int(env.action_space.sample())
        else:
            action = select_greedy_action(policy, observation, torch_device)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.add(observation, action, reward, next_observation, done)
        observation = next_observation
        if done:
            observation, _ = env.reset()

        if step >= config.learning_starts and replay.length >= config.batch_size:
            if step % config.train_frequency == 0:
                loss = _train_one_step(policy, target, replay, optimizer, config, torch_device)
                losses.append(loss)
                updates += 1

        if (step + 1) % config.target_update_interval == 0:
            target.load_state_dict(policy.state_dict())

    return policy, DuelingDQNTrainResult(
        losses=losses,
        final_epsilon=_epsilon_at_step(config, config.total_timesteps - 1),
        updates=updates,
    )


def evaluate_dueling_dqn(
    model: DuelingDQN,
    env: PaperDiscreteMarketMakingEnv,
    *,
    seed: int,
    device: str = "cpu",
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    observation, _ = env.reset(seed=seed)
    done = False
    info: dict[str, object] = {}
    torch_device = torch.device(device)
    model.eval()
    while not done:
        action = select_greedy_action(model, observation, torch_device)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    metrics = info.get("metrics", {})
    trade_log = info.get("trade_log", [])
    if not isinstance(metrics, dict) or not isinstance(trade_log, list):
        raise RuntimeError("D-DQN evaluation did not return terminal metrics")
    return metrics, trade_log


def save_dueling_dqn(
    model: DuelingDQN,
    path: Path,
    *,
    config: DuelingDQNConfig,
    train_result: DuelingDQNTrainResult,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "train_result": asdict(train_result),
        },
        path,
    )


def select_greedy_action(model: DuelingDQN, observation: Observation, device: torch.device) -> int:
    with torch.no_grad():
        q_values = model(_observation_to_tensor(observation, device))
    return int(q_values.argmax(dim=1).item())


def _train_one_step(
    policy: DuelingDQN,
    target: DuelingDQN,
    replay: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    config: DuelingDQNConfig,
    device: torch.device,
) -> float:
    observations, next_observations, actions, rewards, dones = replay.sample(
        config.batch_size, device
    )
    q_values = policy(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = policy(next_observations).argmax(dim=1)
        next_q_values = target(next_observations).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards + config.gamma * (1.0 - dones) * next_q_values

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.detach().cpu())


def _observation_to_tensor(observation: Observation, device: torch.device) -> TensorObservation:
    return {
        key: torch.from_numpy(value).unsqueeze(0).to(device) for key, value in observation.items()
    }


def _epsilon_at_step(config: DuelingDQNConfig, step: int) -> float:
    decay_steps = max(1, int(config.total_timesteps * config.epsilon_fraction))
    fraction = min(1.0, step / decay_steps)
    return float(config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start))


def _validate_lob_mode(lob_mode: str) -> str:
    if lob_mode not in {"attn", "mlp", "none"}:
        raise ValueError("lob_mode must be one of: attn, mlp, none")
    return lob_mode
