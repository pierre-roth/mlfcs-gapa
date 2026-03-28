from __future__ import annotations

import numpy as np
import pandas as pd

from lobmmx.config import RLTrainConfig
from lobmmx.data import DayData
from lobmmx.env import MarketMakingEnv


def _synthetic_day(length: int = 12) -> DayData:
    timestamps = pd.date_range("2026-03-02 10:00:00", periods=length, freq="1s")
    lob = np.zeros((length, 40), dtype=np.float32)
    for level in range(10):
        ask_price = 100.01 + 0.01 * level
        bid_price = 99.99 - 0.01 * level
        lob[:, level * 4 + 0] = ask_price
        lob[:, level * 4 + 1] = 100.0
        lob[:, level * 4 + 2] = bid_price
        lob[:, level * 4 + 3] = 100.0
    midprice = np.full(length, 100.0, dtype=np.float32)
    ask1 = np.full(length, 100.01, dtype=np.float32)
    bid1 = np.full(length, 99.99, dtype=np.float32)
    spread = ask1 - bid1
    dynamic = np.zeros((length, 30), dtype=np.float32)
    handcrafted = np.zeros((length, 10), dtype=np.float32)
    return DayData(
        symbol="AAPL",
        day="20260302",
        timestamps=timestamps,
        lob=lob,
        midprice=midprice,
        ask1=ask1,
        bid1=bid1,
        spread=spread,
        dynamic=dynamic,
        handcrafted=handcrafted,
        trades_by_index={},
        trade_indices=np.array([], dtype=np.int64),
        signed_trade_volume=np.zeros(length, dtype=np.float32),
        normalized_lob=lob.copy(),
    )


def test_lobmmx_defaults_select_on_pnl() -> None:
    cfg = RLTrainConfig(mode="smoke", symbols=["AAPL"]).apply_mode_defaults()
    assert cfg.ppo_selection_metric == "pnl_mean"
    assert cfg.zeta == 0.0
    assert cfg.terminal_inventory_cost_scale == 1.0


def test_lobmmx_deterministic_random_initial_inventory() -> None:
    day = _synthetic_day()
    cfg = RLTrainConfig(
        mode="smoke",
        symbols=["AAPL"],
        lookback=2,
        latency=1,
        episode_length=5,
        random_initial_inventory=True,
        initial_inventory_max=5,
        max_inventory=10,
    ).apply_mode_defaults()
    env_a = MarketMakingEnv(day, cfg, reward_mode=cfg.reward_mode)
    env_b = MarketMakingEnv(day, cfg, reward_mode=cfg.reward_mode)
    span = env_a.available_episodes()[0]
    env_a.set_eval_context(0)
    env_b.set_eval_context(0)
    env_a.reset(span)
    env_b.reset(span)
    assert env_a.initial_inventory == env_b.initial_inventory


def test_lobmmx_trade_inventory_reward_uses_terminal_penalty_only() -> None:
    day = _synthetic_day()
    cfg = RLTrainConfig(
        mode="smoke",
        symbols=["AAPL"],
        lookback=2,
        latency=1,
        episode_length=5,
        random_initial_inventory=False,
        allow_terminal_inventory=True,
        reward_mode="trade_inventory",
        zeta=0.0,
        terminal_inventory_cost_scale=1.0,
    ).apply_mode_defaults()
    env = MarketMakingEnv(day, cfg, reward_mode=cfg.reward_mode)
    span = env.available_episodes()[0]
    env.reset(span)
    env.inventory = 5.0
    env.initial_inventory = 5.0

    rewards = []
    done = False
    while not done:
        _, reward, done, _ = env.step([0.5, 0.5, 0.5])
        rewards.append(reward)

    assert all(abs(value) < 1e-9 for value in rewards[:-1])
    assert rewards[-1] < 0.0
    assert env.trading_pnl == 0.0
    result = env.episode_result("PPO_full", 0)
    assert result.avg_abs_position == 0.0
    assert result.pnl == 0.0
