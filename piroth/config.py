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
        base_depth=4_500,
        depth_decay=0.84,
        default_spread_ticks=2,
        volatility_scale=0.45,
    ),
    "002415": SymbolSpec(
        symbol="002415",
        base_price=32.00,
        tick_size=0.01,
        lot_size=100,
        events_per_day=82_000,
        base_depth=4_500,
        depth_decay=0.83,
        default_spread_ticks=1,
        volatility_scale=1.55,
    ),
    "AAPL": SymbolSpec(
        symbol="AAPL",
        base_price=265.00,
        tick_size=0.01,
        lot_size=100,
        events_per_day=5_500_000,
        base_depth=1_000,
        depth_decay=0.82,
        default_spread_ticks=3,
        volatility_scale=1.0,
    ),
    "GOOGL": SymbolSpec(
        symbol="GOOGL",
        base_price=303.00,
        tick_size=0.01,
        lot_size=100,
        events_per_day=7_900_000,
        base_depth=1_000,
        depth_decay=0.82,
        default_spread_ticks=3,
        volatility_scale=1.0,
    ),
    "GOOG": SymbolSpec(
        symbol="GOOG",
        base_price=303.00,
        tick_size=0.01,
        lot_size=100,
        events_per_day=7_900_000,
        base_depth=1_000,
        depth_decay=0.82,
        default_spread_ticks=3,
        volatility_scale=1.0,
    ),
}


@dataclass
class SimulatorConfig:
    data_dir: str = "data/piroth2"
    output_root: str = "artifacts_piroth2"
    run_name: str = "piroth2_diagnostics"
    mode: str = "medium"
    data_source: str = "synthetic"
    real_data_root: str = "/cluster/work/math/piroth/mlfcs-gapa/data/processed"
    real_start_time: str = "10:00:00"
    real_end_time: str = "15:30:00"
    real_chunk_size: int = 250_000
    real_event_stride: int = 1
    real_build_depth_cube: bool = False
    symbol: str = "000001"
    seed: int = 7
    events_per_day_override: int | None = None

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

    fair_value_vol_ticks: float = 0.12
    fair_value_reversion: float = 0.0040
    anchor_reversion: float = 0.0005
    daily_price_band_ticks: int = 260
    regime_switch_prob: float = 0.0014
    regime_drift_ticks: float = 0.025
    regime_persistence: float = 0.994
    metaorder_start_prob: float = 0.0025
    metaorder_persistence: float = 0.996
    metaorder_drift_ticks: float = 0.120
    shock_prob: float = 0.0009
    shock_size_ticks: float = 5.0
    volatility_cluster_strength: float = 0.0
    volatility_cluster_persistence: float = 0.985
    order_flow_memory: float = 0.0

    noise_market_order_prob: float = 0.33
    informed_market_order_prob: float = 0.20
    liquidity_add_prob: float = 0.24
    cancel_prob: float = 0.13
    mm_refresh_prob: float = 0.12

    touch_join_probability: float = 0.70
    touch_replenish_probability: float = 0.85
    queue_replenish_scale: float = 0.88
    queue_deplete_scale: float = 0.72
    stale_cancel_bias: float = 0.72

    market_order_mean_lots: float = 4.0
    informed_order_scale: float = 2.0
    limit_order_mean_lots: float = 2.4
    cancel_mean_fraction: float = 0.45

    mm_half_spread_ticks: float = 0.70
    mm_inventory_skew_ticks: float = 0.10
    mm_depth_levels: int = 4
    mm_refresh_sensitivity: float = 1.00

    synthetic_market_impact_scale: float = 0.0
    export_day_count: int = 2
    export_depth_radius_ticks: int = 15
    max_pretrain_samples_per_day: int | None = None
    max_train_episodes_per_day: int | None = None
    max_eval_episodes_per_day: int | None = None

    as_gamma: float = 0.08
    as_fill_horizon_events: int = 64
    as_max_distance_ticks: int = 6

    trade_unit_override: int | None = None
    max_inventory_units: int = 10
    max_bias: float = 0.05
    max_spread: float = 0.10
    continuous_action_mode: str = "author"
    matching_mode: str = "author_single"
    reward_mode: str = "author_pnl"
    reward_eta: float = 0.5
    reward_zeta: float = 0.01
    reward_use_dampened_pnl: bool = True
    reward_use_trading_pnl: bool = True
    reward_use_inventory_penalty: bool = True
    reward_spread_penalty_threshold: float = 0.02
    reward_spread_penalty_scale: float = 100.0
    reward_pnl_weight: float = 1.0
    reward_trading_pnl_weight: float = 1.0
    reward_inventory_penalty_weight: float = 1.0
    reward_spread_penalty_weight: float = 1.0
    maker_rebate_per_share: float = 0.0

    pretrain_model_type: str = "attnlob"
    pretrain_horizon: int = 10
    pretrain_threshold: float = 1e-5
    pretrain_stable_windows_only: bool = False
    pretrain_class_weight_mode: str = "none"
    include_lob_state: bool = True
    include_market_state: bool = True
    include_agent_state: bool = True
    author_market_state_alias: bool = False
    torch_batch_size: int = 256
    torch_learning_rate: float = 1e-4
    torch_encoder_learning_rate_scale: float = 1.0
    torch_backbone_learning_rate_scale: float = 1.0
    torch_epochs: int = 5
    ppo_epochs: int = 5
    ppo_rollouts_per_epoch: int | None = None
    ppo_shuffle_episodes: bool = True
    ppo_update_epochs: int = 4
    ppo_clip: float = 0.2
    ppo_entropy_coef: float = 0.01
    ppo_entropy_coef_final: float | None = None
    ppo_value_coef: float = 0.5
    ppo_initial_log_std: float = -1.5
    ppo_initial_spread_bias: float = -0.70
    bc_as_init: bool = False
    bc_as_epochs: int = 2
    bc_as_freeze_backbone: bool = True
    bc_as_freeze_encoder_only: bool = True
    bc_as_max_samples_per_day: int | None = 10_000
    bc_as_loss_weight: float = 1.0
    dqn_replay_size: int = 200_000
    dqn_min_replay: int = 2_000
    dqn_update_interval: int = 4
    dqn_target_update_steps: int = 1_000
    dqn_epsilon_start: float = 0.20
    dqn_epsilon_end: float = 0.02
    dqn_epsilon_decay: float = 0.80
    dqn_discrete_offset_pairs: str = "0:0,0:1,1:0,1:1,0:2,2:0,2:2"
    discount: float = 0.99

    def apply_mode_defaults(self) -> None:
        if self.mode == "smoke":
            self.num_days = min(self.num_days, 4)
            self.train_days = min(self.train_days, 2)
            self.test_days = min(self.test_days, 2)
            self.max_pretrain_samples_per_day = self.max_pretrain_samples_per_day or 2_048
            self.max_train_episodes_per_day = self.max_train_episodes_per_day or 1
            self.max_eval_episodes_per_day = self.max_eval_episodes_per_day or 1
            self.torch_epochs = min(self.torch_epochs, 2)
            self.ppo_epochs = min(self.ppo_epochs, 2)
            self.ppo_update_epochs = min(self.ppo_update_epochs, 1)
            self.ppo_rollouts_per_epoch = self.ppo_rollouts_per_epoch or 2
            self.dqn_replay_size = min(self.dqn_replay_size, 2_000)
            self.dqn_min_replay = min(self.dqn_min_replay, 100)
            self.dqn_update_interval = max(self.dqn_update_interval, 8)
            self.as_fill_horizon_events = min(self.as_fill_horizon_events, 16)
            self.as_max_distance_ticks = min(self.as_max_distance_ticks, 4)
        elif self.mode == "medium":
            self.num_days = min(self.num_days, 8)
            self.train_days = min(self.train_days, 4)
            self.test_days = min(self.test_days, 4)
            self.max_pretrain_samples_per_day = self.max_pretrain_samples_per_day or 50_000
            self.max_train_episodes_per_day = self.max_train_episodes_per_day or 8
            self.max_eval_episodes_per_day = self.max_eval_episodes_per_day or 6
            self.torch_epochs = min(self.torch_epochs, 4)
            self.ppo_epochs = min(self.ppo_epochs, 4)
            self.ppo_update_epochs = min(self.ppo_update_epochs, 2)
            self.ppo_rollouts_per_epoch = self.ppo_rollouts_per_epoch or 8
            self.dqn_replay_size = min(self.dqn_replay_size, 20_000)
            self.dqn_min_replay = min(self.dqn_min_replay, 500)
            self.as_fill_horizon_events = min(self.as_fill_horizon_events, 48)
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

    @property
    def trade_unit(self) -> int:
        if self.trade_unit_override is not None:
            return max(int(self.trade_unit_override), 1)
        return int(self.symbol_spec.lot_size)

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
