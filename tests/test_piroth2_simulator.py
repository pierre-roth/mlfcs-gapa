from __future__ import annotations

from piroth.config import DiagnosticsConfig
from piroth.simulator import SyntheticMarketGenerator


def test_generated_day_has_expected_fileshape() -> None:
    config = DiagnosticsConfig(mode="smoke", symbol="000001", seed=1)
    generator = SyntheticMarketGenerator(config)
    day = generator.generate_day(generator.business_days()[0])

    assert set(["timestamp", "midprice", "ask1_price", "bid1_price"]).issubset(day.price.columns)
    assert set(["timestamp", "ask1_price", "ask1_volume"]).issubset(day.ask.columns)
    assert set(["timestamp", "bid1_price", "bid1_volume"]).issubset(day.bid.columns)
    assert set(["timestamp", "event_type", "agent_type"]).issubset(day.msg.columns)
    assert set(["timestamp", "fair_value", "event_kind"]).issubset(day.latent.columns)
    assert len(day.price) == len(day.ask) == len(day.bid) == len(day.latent)
    assert len(day.msg) >= len(day.price)
    assert day.depth_cube.shape[0] == len(day.price)
