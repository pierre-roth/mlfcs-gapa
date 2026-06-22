import numpy as np
import pytest

from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.extensions.as_behavior_cloning import collect_as_demonstrations
from mlfcs_gapa.extensions.as_guidance import (
    ASGuidanceConfig,
    apply_hard_as_window,
    as_divergence_penalty,
    make_as_strategy,
    paper_action_to_env_action,
)
from mlfcs_gapa.extensions.as_guided_env import ASGuidedMarketMakingEnv


def test_as_teacher_demo_collection_shapes() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=220, seed=501))
    demos = collect_as_demonstrations(
        dataset,
        n_samples=12,
        episode_events=90,
        normalize_actions=True,
        seed=1,
    )

    assert demos.size == 12
    assert demos.actions.shape == (12, 2)
    assert demos.observations["lob_state"].shape[0] == 12
    assert np.all(demos.actions >= -1.0)
    assert np.all(demos.actions <= 1.0)


def test_hard_as_window_clips_paper_actions() -> None:
    clipped = apply_hard_as_window(
        np.array([1.0, 0.0]),
        np.array([0.4, 0.6]),
        hard_window_bias=0.1,
        hard_window_spread=0.2,
    )

    assert np.allclose(clipped, np.array([0.5, 0.4], dtype=np.float32))


def test_soft_as_guidance_subtracts_divergence_penalty() -> None:
    penalty = as_divergence_penalty(
        np.array([0.5, 0.1]),
        np.array([0.5, 0.4]),
        soft_penalty=2.0,
    )

    assert penalty == pytest.approx(0.18)


def test_as_guided_env_logs_teacher_and_raw_actions() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=220, seed=502))
    strategy = make_as_strategy(dataset, episode_events=90)
    env = ASGuidedMarketMakingEnv(
        dataset,
        as_strategy=strategy,
        guidance=ASGuidanceConfig(mode="hard", hard_window_bias=0.05, hard_window_spread=0.05),
        episode_events=90,
        normalize_actions=True,
        seed=1,
    )
    env.reset()
    _, _, _, _, info = env.step(paper_action_to_env_action(np.array([1.0, 1.0]), normalize_actions=True))

    row = env.trade_log[-1]
    assert "teacher_action_bias" in row
    assert "teacher_action_spread" in row
    assert abs(row["action_bias"] - row["teacher_action_bias"]) <= 0.050001
    assert abs(row["action_spread"] - row["teacher_action_spread"]) <= 0.050001
    assert "raw_paper_action" in info


def test_soft_as_guided_env_uses_profit_reward_base() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=220, seed=503))
    strategy = make_as_strategy(dataset, episode_events=90)
    env = ASGuidedMarketMakingEnv(
        dataset,
        as_strategy=strategy,
        guidance=ASGuidanceConfig(mode="soft", soft_penalty=0.5, base_reward="profit"),
        episode_events=90,
        normalize_actions=True,
        seed=1,
    )
    env.reset()
    _, reward, _, _, info = env.step(
        paper_action_to_env_action(np.array([1.0, 0.0]), normalize_actions=True)
    )

    row = env.trade_log[-1]
    assert row["base_reward"] == "profit"
    assert reward == pytest.approx(row["profit_reward"] - row["as_guidance_penalty"])
    assert info["base_reward"] == "profit"
