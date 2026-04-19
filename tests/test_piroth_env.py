from __future__ import annotations

from pathlib import Path

from piroth.config import GenerateConfig, TrainConfig
from piroth.data import load_day
from piroth.env import ContinuousMarketEnv, Fill
from piroth.simulator import generate_dataset


def _load_env(tmp_path: Path) -> ContinuousMarketEnv:
    data_dir = tmp_path / "sim"
    gen_cfg = GenerateConfig(mode="smoke", data_dir=str(data_dir), symbols=["000001"], seed=5).apply_mode_defaults()
    generate_dataset(gen_cfg)
    train_cfg = TrainConfig(mode="smoke", data_dir=str(data_dir), symbols=["000001"], seed=5).apply_mode_defaults()
    day = load_day("000001", "20191101", train_cfg)
    return ContinuousMarketEnv(day, train_cfg)


def test_hybrid_reward_matches_formula(tmp_path: Path) -> None:
    env = _load_env(tmp_path)
    span = env.available_episodes()[0]
    env.reset(span)
    env.prev_value = 0.2
    env.value = 0.5
    env.inventory = 200.0
    reward = env._reward([Fill(price=10.0, volume=100.0)], mid=10.02)
    expected = (0.5 - 0.2) - max(0.0, env.config.eta * (0.5 - 0.2)) + 100.0 * (10.02 - 10.0) - env.config.zeta * (env.inventory / env.config.trade_unit) ** 2
    assert abs(reward - expected) < 1e-8


def test_terminal_liquidation_flattens_inventory(tmp_path: Path) -> None:
    env = _load_env(tmp_path)
    span = env.available_episodes()[0]
    obs = env.reset(span)
    done = False
    while not done:
        obs, _, done, _ = env.step([0.5, 0.5])
    assert env.inventory == 0.0
    assert obs.flat.shape[0] == 48


def test_maker_rebate_only_applies_to_passive_fills(tmp_path: Path) -> None:
    env = _load_env(tmp_path)
    env.config.use_maker_rebate = True
    env.config.maker_rebate_per_share = 0.002
    env.reset(env.available_episodes()[0])
    env._apply_fill(Fill(price=10.0, volume=100.0, taker=False))
    assert abs(env.cash - (-1000.0 + 0.2)) < 1e-8
    env._apply_fill(Fill(price=10.0, volume=-100.0, taker=True))
    assert abs(env.cash - 0.2) < 1e-8
