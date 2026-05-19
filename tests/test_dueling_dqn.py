import torch

from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.env.discrete_env import PaperDiscreteMarketMakingEnv
from mlfcs_gapa.training.dueling_dqn import DuelingDQN, DuelingDQNConfig, train_dueling_dqn


def test_dueling_dqn_outputs_one_q_value_per_discrete_action() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=96))
    env = PaperDiscreteMarketMakingEnv(dataset, episode_events=120, seed=1)
    observation, _ = env.reset()
    model = DuelingDQN(env.observation_space, int(env.action_space.n), features_dim=32)

    batch = {key: torch.from_numpy(value).unsqueeze(0) for key, value in observation.items()}
    q_values = model(batch)

    assert tuple(q_values.shape) == (1, 8)
    assert torch.isfinite(q_values).all()


def test_train_dueling_dqn_smoke_updates_q_network() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=97))
    env = PaperDiscreteMarketMakingEnv(dataset, episode_events=100, seed=1)
    config = DuelingDQNConfig(
        total_timesteps=12,
        learning_starts=4,
        buffer_size=64,
        batch_size=4,
        target_update_interval=8,
        features_dim=16,
        seed=5,
    )

    _, result = train_dueling_dqn(env, config=config)

    assert result.updates > 0
    assert len(result.losses) == result.updates


def test_dueling_dqn_supports_paper_ablation_modes() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=98))
    env = PaperDiscreteMarketMakingEnv(dataset, episode_events=120, seed=1)
    observation, _ = env.reset()
    batch = {key: torch.from_numpy(value).unsqueeze(0) for key, value in observation.items()}

    mlp_model = DuelingDQN(
        env.observation_space,
        int(env.action_space.n),
        features_dim=16,
        lob_mode="mlp",
        use_dynamic_state=False,
        use_agent_state=True,
    )
    no_lob_model = DuelingDQN(
        env.observation_space,
        int(env.action_space.n),
        features_dim=16,
        lob_mode="none",
        use_dynamic_state=True,
        use_agent_state=True,
    )

    assert tuple(mlp_model(batch).shape) == (1, 8)
    assert tuple(no_lob_model(batch).shape) == (1, 8)
