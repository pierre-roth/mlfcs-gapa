import numpy as np
import pytest

from mlfcs_gapa.env.actions import continuous_action_to_quote
from mlfcs_gapa.env.rewards import dampened_pnl, hybrid_reward, inventory_penalty, trading_pnl


def test_continuous_action_quote_is_inventory_skewed() -> None:
    quote_long = continuous_action_to_quote(np.array([1.0, 0.5]), mid_price=16.45, inventory=100)
    quote_short = continuous_action_to_quote(np.array([1.0, 0.5]), mid_price=16.45, inventory=-100)
    quote_flat = continuous_action_to_quote(np.array([1.0, 0.5]), mid_price=16.45, inventory=0)

    assert quote_long.reservation_price == pytest.approx(16.40)
    assert quote_short.reservation_price == pytest.approx(16.50)
    assert quote_flat.reservation_price == pytest.approx(16.45)
    assert quote_long.ask_price >= quote_long.bid_price
    assert quote_long.ask_volume == -100
    assert quote_long.bid_volume == 100


def test_continuous_action_rejects_non_paper_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        continuous_action_to_quote(np.array([-0.1, 0.5]), mid_price=10.0, inventory=0)


def test_reward_components_match_paper_equations() -> None:
    assert dampened_pnl(10.0, eta=0.5) == pytest.approx(5.0)
    assert dampened_pnl(-10.0, eta=0.5) == pytest.approx(-10.0)
    assert trading_pnl(mid_price=10.0, trade_price=9.99, trade_volume=100) == pytest.approx(1.0)
    assert trading_pnl(mid_price=10.0, trade_price=10.01, trade_volume=-100) == pytest.approx(1.0)
    assert inventory_penalty(1_000, zeta=0.01) == pytest.approx(1.0)


def test_hybrid_reward_is_dp_plus_tp_minus_ip() -> None:
    breakdown = hybrid_reward(
        delta_pnl=10.0,
        mid_price=10.0,
        trade_price=9.99,
        trade_volume=100,
        inventory=100,
    )

    assert breakdown.dampened_pnl == pytest.approx(5.0)
    assert breakdown.trading_pnl == pytest.approx(1.0)
    assert breakdown.inventory_penalty == pytest.approx(0.01)
    assert breakdown.reward == pytest.approx(5.99)
