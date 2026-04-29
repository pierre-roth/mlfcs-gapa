from __future__ import annotations

import numpy as np
import pandas as pd

from piroth.config import DiagnosticsConfig
from piroth.paper_env import PaperAction, PaperTradingEnv
from piroth.paper_policies import ContinuousActionPolicy, DiscreteActionPolicy
from piroth.simulator import SyntheticDay


def test_two_sided_fill_updates_cash_turnover_and_fill_step_once() -> None:
    day = _minimal_day()
    config = DiagnosticsConfig(
        mode="smoke",
        lookback=1,
        latency=0,
        episode_length=2,
        stable_windows=["10:00:00-10:01:00"],
        max_eval_episodes_per_day=1,
        matching_mode="multi_fill",
        reward_mode="hybrid",
    )
    env = PaperTradingEnv(day, config, episode_start=0, episode_stop=2, episode_index=0)
    env.reset()

    result = env.step(PaperAction(ask_price=10.01, ask_volume=-100, bid_price=9.99, bid_volume=100))

    assert result.info["trade_volume"] == 0
    assert env.inventory == 0
    assert env.value == 2.0
    assert env.turnover == 2000.0
    assert env.trades == 2
    assert env.fill_steps == 1


def test_author_matching_keeps_one_net_fill_per_step() -> None:
    day = _minimal_day()
    config = DiagnosticsConfig(
        mode="smoke",
        lookback=1,
        latency=0,
        episode_length=2,
        stable_windows=["10:00:00-10:01:00"],
        matching_mode="author_single",
    )
    env = PaperTradingEnv(day, config, episode_start=0, episode_stop=2, episode_index=0)
    env.reset()

    fills = env.match(PaperAction(ask_price=10.01, ask_volume=-100, bid_price=9.99, bid_volume=100), event_idx=1)

    assert fills == [(9.99, 100)]


def test_author_reward_is_pnl_minus_spread_penalty() -> None:
    day = _minimal_day()
    config = DiagnosticsConfig(
        mode="smoke",
        lookback=1,
        latency=0,
        episode_length=2,
        stable_windows=["10:00:00-10:01:00"],
        reward_mode="author_pnl",
    )
    env = PaperTradingEnv(day, config, episode_start=0, episode_stop=2, episode_index=0)
    env.reset()
    env.previous_value = 2.0
    env.value = 5.0

    reward = env.reward(0.0, 0, PaperAction(ask_price=10.04, ask_volume=-100, bid_price=9.99, bid_volume=100))

    assert np.isclose(reward, 3.0 - 5.0)


def test_hybrid_reward_uses_linear_component_weights() -> None:
    day = _minimal_day()
    config = DiagnosticsConfig(
        mode="smoke",
        lookback=1,
        latency=0,
        episode_length=2,
        stable_windows=["10:00:00-10:01:00"],
        reward_mode="hybrid",
        reward_use_dampened_pnl=False,
        reward_use_trading_pnl=True,
        reward_use_inventory_penalty=True,
        reward_spread_penalty_scale=10.0,
        reward_pnl_weight=0.5,
        reward_trading_pnl_weight=2.0,
        reward_inventory_penalty_weight=3.0,
        reward_spread_penalty_weight=0.25,
        reward_zeta=0.1,
    )
    env = PaperTradingEnv(day, config, episode_start=0, episode_stop=2, episode_index=0)
    env.reset()
    env.previous_value = 2.0
    env.value = 6.0
    env.inventory = config.trade_unit

    reward = env.reward(
        10.0,
        config.trade_unit,
        PaperAction(ask_price=10.04, ask_volume=-config.trade_unit, bid_price=9.99, bid_volume=config.trade_unit),
        matched_pnl=1.5,
    )

    assert np.isclose(reward, 0.5 * 4.0 + 2.0 * 1.5 - 3.0 * 0.1)


def test_maker_rebate_is_added_to_passive_fill_pnl() -> None:
    day = _minimal_day()
    config = DiagnosticsConfig(
        mode="smoke",
        lookback=1,
        latency=0,
        episode_length=2,
        stable_windows=["10:00:00-10:01:00"],
        max_eval_episodes_per_day=1,
        matching_mode="multi_fill",
        reward_mode="hybrid",
        reward_use_dampened_pnl=False,
        reward_use_trading_pnl=True,
        maker_rebate_per_share=0.0015,
    )
    env = PaperTradingEnv(day, config, episode_start=0, episode_stop=2, episode_index=0)
    env.reset()

    result = env.step(PaperAction(ask_price=10.01, ask_volume=-100, bid_price=9.99, bid_volume=100))

    assert result.info["trade_volume"] == 0
    assert np.isclose(env.value, 2.3)
    assert result.reward > 2.0


def test_one_sided_inventory_guard_does_not_create_artificial_spread() -> None:
    day = _minimal_day()
    config = DiagnosticsConfig(
        mode="smoke",
        lookback=1,
        latency=0,
        episode_length=2,
        stable_windows=["10:00:00-10:01:00"],
        max_inventory_units=1,
    )
    env = PaperTradingEnv(day, config, episode_start=0, episode_stop=2, episode_index=0)
    env.reset()
    env.inventory = config.symbol_spec.lot_size

    result = env.step(PaperAction(ask_price=10.01, ask_volume=-100, bid_price=9.99, bid_volume=100))

    assert result.terminal
    assert max(env.spread_path) == config.symbol_spec.tick_size


def test_continuous_policy_can_quote_two_ticks() -> None:
    day = _minimal_day()
    config = DiagnosticsConfig(
        mode="smoke",
        lookback=1,
        latency=0,
        episode_length=2,
        stable_windows=["10:00:00-10:01:00"],
    )
    env = PaperTradingEnv(day, config, episode_start=0, episode_stop=2, episode_index=0)
    env.reset()

    action = ContinuousActionPolicy([0.0, -0.70]).act(env.state(), env)

    assert action.ask_price - action.bid_price <= 0.02


def test_discrete_policy_matches_author_inventory_limit() -> None:
    day = _minimal_day()
    config = DiagnosticsConfig(
        mode="smoke",
        lookback=1,
        latency=0,
        episode_length=2,
        stable_windows=["10:00:00-10:01:00"],
        max_inventory_units=1,
    )
    env = PaperTradingEnv(day, config, episode_start=0, episode_stop=2, episode_index=0)
    env.reset()
    env.inventory = -2 * config.symbol_spec.lot_size

    action = DiscreteActionPolicy(0).act(env.state(), env)

    assert action.ask_volume == 0
    assert action.bid_volume == config.symbol_spec.lot_size


def test_trade_unit_override_changes_policy_size_and_inventory_limit() -> None:
    day = _minimal_day()
    config = DiagnosticsConfig(
        mode="smoke",
        lookback=1,
        latency=0,
        episode_length=2,
        stable_windows=["10:00:00-10:01:00"],
        max_inventory_units=1,
        trade_unit_override=1,
    )
    env = PaperTradingEnv(day, config, episode_start=0, episode_stop=2, episode_index=0)
    env.reset()
    env.inventory = -2

    action = DiscreteActionPolicy(0).act(env.state(), env)

    assert action.ask_volume == 0
    assert action.bid_volume == 1


def test_discrete_policy_can_use_configured_wider_quote_offsets() -> None:
    day = _minimal_day()
    config = DiagnosticsConfig(
        mode="smoke",
        lookback=1,
        latency=0,
        episode_length=2,
        stable_windows=["10:00:00-10:01:00"],
        dqn_discrete_offset_pairs="1:1,1:2,2:1,2:2,1:3,3:1,3:3",
    )
    env = PaperTradingEnv(day, config, episode_start=0, episode_stop=2, episode_index=0)
    env.reset()

    action = DiscreteActionPolicy(0).act(env.state(), env)

    assert np.isclose(action.ask_price, 10.02)
    assert np.isclose(action.bid_price, 9.98)


def test_author_raw_continuous_policy_keeps_literal_reference_action_scale() -> None:
    day = _minimal_day()
    config = DiagnosticsConfig(
        mode="smoke",
        lookback=1,
        latency=0,
        episode_length=2,
        stable_windows=["10:00:00-10:01:00"],
        continuous_action_mode="author_raw",
    )
    env = PaperTradingEnv(day, config, episode_start=0, episode_stop=2, episode_index=0)
    env.reset()

    action = ContinuousActionPolicy([0.0, 0.10]).act(env.state(), env)

    assert np.isclose(action.ask_price - action.bid_price, 0.02)


def _minimal_day() -> SyntheticDay:
    timestamps = pd.to_datetime(["2019-11-01 10:00:00", "2019-11-01 10:00:01"])
    ask = pd.DataFrame([_lob_row(ts, "ask", 10.01, 1000) for ts in timestamps])
    bid = pd.DataFrame([_lob_row(ts, "bid", 9.99, 1000) for ts in timestamps])
    price = pd.DataFrame(
        {
            "timestamp": timestamps,
            "midprice": [10.0, 10.0],
            "ask1_price": [10.01, 10.01],
            "bid1_price": [9.99, 9.99],
            "spread_ticks": [2, 2],
            "return_bp": [0.0, 0.0],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": [timestamps[1], timestamps[1]],
            "price": [10.02, 9.98],
            "size": [100, 100],
            "aggressor_side": ["B", "A"],
        }
    )
    msg = pd.DataFrame(
        {
            "timestamp": timestamps,
            "market_buy_volume": [0, 100],
            "market_buy_n": [0, 1],
            "market_sell_volume": [0, 100],
            "market_sell_n": [0, 1],
        }
    )
    latent = pd.DataFrame({"timestamp": timestamps, "fair_value": [10.0, 10.0], "event_kind": ["test", "test"]})
    return SyntheticDay(
        symbol="000001",
        day="20191101",
        ask=ask,
        bid=bid,
        price=price,
        trades=trades,
        msg=msg,
        event_log=pd.DataFrame(),
        latent=latent,
        depth_cube=np.zeros((2, 31), dtype=np.float32),
    )


def _lob_row(timestamp: pd.Timestamp, side: str, start_price: float, volume: int) -> dict[str, float | int | pd.Timestamp]:
    row: dict[str, float | int | pd.Timestamp] = {"timestamp": timestamp}
    sign = 1 if side == "ask" else -1
    for level in range(1, 11):
        row[f"{side}{level}_price"] = start_price + sign * (level - 1) * 0.01
        row[f"{side}{level}_volume"] = volume
    return row
