"""Paper constants that should not silently drift during replication.

These values come from the local paper source in ``paper/paper.tex``. They are
kept in one module so experiments can state clearly when they are paper-faithful
and when they intentionally deviate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperMarketConfig:
    """Market and episode constants stated in the paper."""

    lob_levels: int = 10
    lob_features_per_level: int = 4
    window_length: int = 50
    midprice_horizon_events: int = 10
    midprice_label_threshold: float = 1e-5
    minimum_trade_unit: int = 100
    omega_inventory_units: int = 10
    episode_events: int = 2_000
    eta_dampened_pnl: float = 0.5
    zeta_inventory_penalty: float = 0.01
    max_bias: float = 0.05
    max_spread: float = 0.1
    transaction_cost: float = 0.0

    @property
    def lob_width(self) -> int:
        return self.lob_levels * self.lob_features_per_level

    @property
    def max_inventory(self) -> int:
        return self.omega_inventory_units * self.minimum_trade_unit

    @property
    def lob_window_shape(self) -> tuple[int, int]:
        return (self.window_length, self.lob_width)


PAPER = PaperMarketConfig()


PAPER_STOCKS: dict[str, str] = {
    "000001": "Ping An Bank Co., Ltd.",
    "000858": "Wuliangye Yibin Co., Ltd.",
    "002415": "Hikvision Technology Co., Ltd.",
}


PAPER_TRADING_DAYS_201911: tuple[str, ...] = (
    "2019-11-01",
    "2019-11-04",
    "2019-11-05",
    "2019-11-06",
    "2019-11-07",
    "2019-11-08",
    "2019-11-11",
    "2019-11-12",
    "2019-11-13",
    "2019-11-14",
    "2019-11-15",
    "2019-11-18",
    "2019-11-19",
    "2019-11-20",
    "2019-11-21",
    "2019-11-22",
    "2019-11-25",
    "2019-11-26",
    "2019-11-27",
    "2019-11-28",
    "2019-11-29",
)


PAPER_PRETRAIN_WINDOWS: tuple[tuple[str, str], ...] = (
    ("10:00:00", "11:30:00"),
    ("13:00:00", "14:30:00"),
)
