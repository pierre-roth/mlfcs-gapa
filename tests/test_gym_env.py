import numpy as np

from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.env.gym_env import PaperMarketMakingEnv
from mlfcs_gapa.paper.constants import PAPER


def test_paper_market_making_env_reset_observation_shapes() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=300, seed=61))
    env = PaperMarketMakingEnv(dataset, episode_events=200, latency_events=1)

    obs, info = env.reset()

    assert info["current_index"] == PAPER.window_length
    assert obs["lob_state"].shape == PAPER.lob_window_shape
    assert obs["dynamic_state"].shape == (24,)
    assert obs["agent_state"].shape == (2,)
    assert env.observation_space.contains(obs)


def test_paper_market_making_env_steps_to_metrics() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=62))
    env = PaperMarketMakingEnv(dataset, episode_events=70, latency_events=1)
    obs, _ = env.reset()
    terminated = False
    steps = 0
    info = {}
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
        assert not truncated
        assert isinstance(reward, float)
        assert env.observation_space.contains(obs)
        steps += 1

    assert steps > 0
    assert "metrics" in info
    assert "pnl" in info["metrics"]
    assert env.account.inventory == 0


def test_paper_market_making_env_can_normalize_ppo_actions() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=63))
    env = PaperMarketMakingEnv(
        dataset,
        episode_events=70,
        latency_events=1,
        normalize_actions=True,
    )
    env.reset()

    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert np.allclose(env.action_space.low, np.array([-1.0, -1.0], dtype=np.float32))
    assert info["paper_action"] == [0.5, 0.5]
    assert env.trade_log[-1]["action_bias"] == 0.5
    assert env.trade_log[-1]["action_spread"] == 0.5


def test_paper_market_making_env_can_sample_random_episode_starts() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=400, seed=64))
    env = PaperMarketMakingEnv(dataset, episode_events=100, random_episode_starts=True)

    starts = []
    for seed in range(5):
        env.reset(seed=seed)
        starts.append(env.episode_start)

    assert len(set(starts)) > 1
    assert all(0 <= start <= 299 for start in starts)

    env.reset(seed=1, options={"episode_start": 10})
    assert env.episode_start == 10
