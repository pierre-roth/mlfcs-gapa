from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    base_price: float
    tick_size: float
    lot_size: int
    events_per_day: int
    base_depth: int
    depth_decay: float
    default_spread_ticks: int
    volatility_scale: float


DEFAULT_SYMBOLS: dict[str, SymbolSpec] = {
    "000001": SymbolSpec(
        symbol="000001",
        base_price=12.50,
        tick_size=0.01,
        lot_size=100,
        events_per_day=105_000,
        base_depth=5_000,
        depth_decay=0.82,
        default_spread_ticks=1,
        volatility_scale=1.0,
    ),
    "000858": SymbolSpec(
        symbol="000858",
        base_price=135.00,
        tick_size=0.01,
        lot_size=100,
        events_per_day=88_000,
        base_depth=2_500,
        depth_decay=0.84,
        default_spread_ticks=2,
        volatility_scale=0.9,
    ),
    "002415": SymbolSpec(
        symbol="002415",
        base_price=32.00,
        tick_size=0.01,
        lot_size=100,
        events_per_day=82_000,
        base_depth=3_200,
        depth_decay=0.83,
        default_spread_ticks=1,
        volatility_scale=1.1,
    ),
}


@dataclass
class SimulatorConfig:
    data_dir: str = "data/piroth2"
    output_root: str = "artifacts_piroth2"
    run_name: str = "piroth2_diagnostics"
    mode: str = "medium"
    symbol: str = "000001"
    seed: int = 7

    num_days: int = 21
    train_days: int = 10
    test_days: int = 11
    lookback: int = 50
    latency: int = 1
    episode_length: int = 2000
    levels: int = 10
    stable_windows: list[str] = field(default_factory=lambda: ["10:00:00-11:30:00", "13:00:00-14:30:00"])
    session_windows: list[str] = field(default_factory=lambda: ["09:30:00-11:30:00", "13:00:00-15:00:00"])

    competing_mm_count: int = 6
    liquidity_provider_count: int = 10
    noise_taker_count: int = 64
    informed_taker_count: int = 10

    target_episode_minutes: float = 4.5
    timestamp_jitter_fraction: float = 0.35

    fair_value_vol_ticks: float = 0.05
    fair_value_reversion: float = 0.0015
    regime_switch_prob: float = 0.0015
    regime_drift_ticks: float = 0.015
    regime_persistence: float = 0.997
    metaorder_start_prob: float = 0.0025
    metaorder_persistence: float = 0.996
    metaorder_drift_ticks: float = 0.07
    shock_prob: float = 0.0008
    shock_size_ticks: float = 4.0

    noise_market_order_prob: float = 0.31
    informed_market_order_prob: float = 0.18
    liquidity_add_prob: float = 0.22
    cancel_prob: float = 0.15
    mm_refresh_prob: float = 0.14

    touch_join_probability: float = 0.55
    touch_replenish_probability: float = 0.50
    queue_replenish_scale: float = 0.85
    queue_deplete_scale: float = 0.65
    stale_cancel_bias: float = 0.65

    market_order_mean_lots: float = 1.8
    informed_order_scale: float = 1.6
    limit_order_mean_lots: float = 2.0
    cancel_mean_fraction: float = 0.35

    mm_half_spread_ticks: float = 0.85
    mm_inventory_skew_ticks: float = 0.10
    mm_depth_levels: int = 4
    mm_refresh_sensitivity: float = 0.85

    synthetic_market_impact_scale: float = 0.0
    export_day_count: int = 2
    export_depth_radius_ticks: int = 15

    as_gamma: float = 0.08
    as_fill_horizon_events: int = 64
    as_max_distance_ticks: int = 6

    def apply_mode_defaults(self) -> None:
        if self.mode == "smoke":
            self.num_days = min(self.num_days, 4)
            self.train_days = min(self.train_days, 2)
            self.test_days = min(self.test_days, 2)
        elif self.mode == "medium":
            self.num_days = min(self.num_days, 8)
            self.train_days = min(self.train_days, 4)
            self.test_days = min(self.test_days, 4)
        elif self.mode == "full":
            return
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def symbol_spec(self) -> SymbolSpec:
        try:
            return DEFAULT_SYMBOLS[self.symbol]
        except KeyError as exc:
            raise KeyError(f"Unsupported symbol {self.symbol!r}") from exc

    def output_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    def export_dir(self) -> Path:
        return self.output_dir() / "exported_days"


@dataclass
class DiagnosticsConfig(SimulatorConfig):
    output_root: str = "artifacts_piroth2"
    run_name: str = "piroth2_diagnostics"
    create_plots: bool = True
    export_generated_days: bool = True
    sample_episode_index: int = 1
    random_window_count: int = 12
    fixed_level_baseline: int = 1
