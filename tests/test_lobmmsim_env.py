from __future__ import annotations

from pathlib import Path

from lobmmsim.config import GenerateConfig, RLTrainConfig
from lobmmsim.data import apply_lob_normalizer, fit_lob_normalizer, load_day_data
from lobmmsim.env import Fill, MarketMakingEnv
from lobmmsim.simulator import generate_dataset


def _load_env(tmp_path: Path) -> MarketMakingEnv:
    data_dir = tmp_path / "sim"
    gen_cfg = GenerateConfig(mode="smoke", data_dir=str(data_dir), symbols=["000001"], seed=5).apply_mode_defaults()
    generate_dataset(gen_cfg)
    rl_cfg = RLTrainConfig(mode="smoke", data_dir=str(data_dir), symbols=["000001"], seed=5).apply_mode_defaults()
    day = load_day_data("000001", "20191101", rl_cfg)
    normalizer = fit_lob_normalizer([day])
    apply_lob_normalizer(day, normalizer)
    return MarketMakingEnv(day, rl_cfg)


def test_paper_reward_matches_components(tmp_path: Path) -> None:
    env = _load_env(tmp_path)
    span = env.available_episodes()[0]
    env.reset(span)
    env.prev_value = 0.2
    env.value = 0.5
    env.inventory = 200.0
    reward = env._reward([Fill(price=10.0, volume=100.0)], midprice=10.02)
    expected = (0.5 - 0.2) - max(0.0, env.config.eta * (0.5 - 0.2)) + 100.0 * (10.02 - 10.0) - env.config.zeta * (env.inventory / env.config.trade_unit) ** 2
    assert abs(reward - expected) < 1e-8


def test_mm_only_reward_ignores_mark_to_market(tmp_path: Path) -> None:
    env = _load_env(tmp_path)
    env.config.reward_mode = "mm_only"
    env.config.inventory_carry_penalty = 0.02
    span = env.available_episodes()[0]
    env.reset(span)
    env.prev_value = 0.2
    env.value = 10.5
    env.inventory = 200.0
    reward = env._reward([Fill(price=10.0, volume=100.0)], midprice=10.02)
    expected = (
        100.0 * (10.02 - 10.0)
        - env.config.zeta * (env.inventory / env.config.trade_unit) ** 2
        - env.config.inventory_carry_penalty * abs(env.inventory) / env.config.trade_unit
    )
    assert abs(reward - expected) < 1e-8


def test_signed_absolute_action_can_move_reservation_from_flat_inventory(tmp_path: Path) -> None:
    env = _load_env(tmp_path)
    env.config.action_mode = "signed_absolute"
    span = env.available_episodes()[0]
    env.reset(span)
    quote_idx = int(env.episode_decisions[env.step_cursor] - env.config.latency)
    mid = float(env.day.midprice[quote_idx])
    orders = env.action_to_orders([1.0, 0.5], quote_idx)
    assert orders["reservation"] > mid


def test_terminal_liquidation_flattens_inventory(tmp_path: Path) -> None:
    env = _load_env(tmp_path)
    span = env.available_episodes()[0]
    obs = env.reset(span)
    done = False
    while not done:
        obs, _, done, _ = env.step([0.5, 0.5])
    assert env.inventory == 0.0
    assert obs.flat.shape[0] == env.day.dynamic.shape[1] + env.day.agent_template.shape[1]
