from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_dir: str = "data/piroth_simulated"
    output_root: str = "artifacts_piroth"
    run_name: str = "piroth_run"
    mode: str = "full"
    symbols: list[str] = field(default_factory=lambda: ["000001", "000858", "002415"])
    seed: int = 7
    device: str = "auto"

    num_days: int = 21
    train_days: int = 8
    val_days: int = 2
    test_days: int = 11
    session_windows: list[str] = field(default_factory=lambda: ["09:30:00-11:30:00", "13:00:00-15:00:00"])
    stable_windows: list[str] = field(default_factory=lambda: ["10:00:00-11:30:00", "13:00:00-14:30:00"])
    use_stable_hours: bool = True

    lookback: int = 50
    latency: int = 1
    trade_unit: int = 100
    tick_size: float = 0.01
    max_inventory_units: int = 10
    episode_length: int = 2000

    pretrain_horizon: int = 10
    pretrain_alpha: float = 1e-5
    pretrain_label_source: str = "price"  # "price" or "signal"
    pretrain_signal_threshold: float = 0.1
    pretrain_backbone: str = "attn"
    pretrain_epochs: int = 10
    pretrain_batch_size: int = 128
    pretrain_lr: float = 1e-3
    pretrain_weight_decay: float = 1e-4
    pretrain_label_smoothing: float = 0.1
    pretrain_num_workers: int = 0
    backbone_name: str = "attn_lob.pt"
    backbone_trainable: bool = True

    max_bias: float = 0.05
    max_spread: float = 0.1
    eta: float = 0.5
    zeta: float = 0.01
    dampened_pnl_weight: float = 1.0
    trade_reward_weight: float = 1.0
    inventory_penalty_weight: float = 1.0
    zero_transaction_cost: bool = True
    use_maker_rebate: bool = False
    maker_rebate_per_share: float = 0.0020
    deterministic_evaluation: bool = True
    eval_seed_base: int = 20260419

    ppo_epochs: int = 6
    ppo_rollouts_per_epoch: int = 8
    ppo_updates: int = 2
    ppo_minibatch_size: int = 256
    ppo_lr: float = 3e-4
    ppo_clip: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    gradient_clip_norm: float = 1.0
    max_train_episodes_per_day: int | None = None
    max_eval_episodes_per_day: int | None = None
    ppo_select_best_model: bool = True
    ppo_selection_metric: str = "pnl_mean"

    rv_windows_s: list[int] = field(default_factory=lambda: [300, 600, 1800])
    rsi_windows_s: list[int] = field(default_factory=lambda: [300, 600, 1800])
    osi_windows_s: list[int] = field(default_factory=lambda: [10, 60, 300])

    events_per_day: dict[str, int] = field(
        default_factory=lambda: {
            "000001": 120_000,
            "000858": 90_000,
            "002415": 60_000,
        }
    )
    base_prices: dict[str, float] = field(
        default_factory=lambda: {
            "000001": 12.5,
            "000858": 135.0,
            "002415": 32.0,
        }
    )
    alpha_signal_scale: float = 1.0
    price_noise_scale: float = 0.004
    market_order_impact_scale: float = 0.95
    flow_reversion_scale: float = 1.0
    noise_taker_rate_scale: float = 1.0
    informed_taker_rate_scale: float = 1.0
    maker_add_rate_scale: float = 1.0
    maker_cancel_rate_scale: float = 1.0
    liquidity_refill_rate_scale: float = 1.0
    maker_join_touch_prob_shift: float = 0.0
    market_order_alpha_sensitivity: float = 0.14
    market_order_imbalance_sensitivity: float = 0.08
    market_order_flow_sensitivity: float = 0.35
    limit_alpha_sensitivity: float = 0.10
    cancel_alpha_sensitivity: float = 0.08
    spread_widen_prob: float = 0.35
    spread_imbalance_threshold: float = 0.3
    spread_alpha_threshold: float = 0.18
    recenter_follow_scale: float = 0.85
    recenter_base_prob: float = 0.06
    recenter_gap_scale: float = 0.12
    recenter_alpha_scale: float = 0.05
    market_order_tick_impact: float = 0.0015
    market_order_alpha_impact: float = 0.0004
    touch_replenish_fraction: float = 0.6

    signal_threshold_for_lob_leak: float = 0.5
    lob_leak_strength: float = 0.3
    informed_hawkes_alpha: float = 0.04
    informed_hawkes_decay: float = 0.97

    # Regime persistence: how long (events) before regime can switch, and per-event prob.
    regime_min_duration: int = 2000
    regime_switch_prob: float = 0.001

    def apply_mode_defaults(self) -> "ExperimentConfig":
        if self.mode == "smoke":
            self.num_days = 4
            self.train_days = 2
            self.val_days = 1
            self.test_days = 1
            self.events_per_day = {symbol: min(self.events_per_day.get(symbol, 60_000), 500) for symbol in self.symbols}
            self.pretrain_epochs = min(self.pretrain_epochs, 1)
            self.pretrain_batch_size = min(self.pretrain_batch_size, 64)
            self.ppo_epochs = min(self.ppo_epochs, 1)
            self.ppo_rollouts_per_epoch = min(self.ppo_rollouts_per_epoch, 2)
            self.ppo_updates = min(self.ppo_updates, 1)
            self.ppo_minibatch_size = min(self.ppo_minibatch_size, 64)
            self.max_train_episodes_per_day = 1
            self.max_eval_episodes_per_day = 1
        elif self.mode == "medium":
            self.num_days = min(self.num_days, 8)
            self.train_days = 4
            self.val_days = 1
            self.test_days = 3
            self.events_per_day = {symbol: min(self.events_per_day.get(symbol, 60_000), 20_000) for symbol in self.symbols}
            self.pretrain_epochs = min(self.pretrain_epochs, 4)
            self.ppo_epochs = min(self.ppo_epochs, 4)
            self.ppo_rollouts_per_epoch = min(self.ppo_rollouts_per_epoch, 6)
            self.max_train_episodes_per_day = 6
            self.max_eval_episodes_per_day = 3
        self.device = self._resolve_device()
        return self

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def output_dir(self) -> Path:
        return Path(self.output_root) / self.run_name


@dataclass
class GenerateConfig(ExperimentConfig):
    overwrite: bool = False


@dataclass
class PretrainConfig(ExperimentConfig):
    pass


@dataclass
class TrainConfig(ExperimentConfig):
    pass


@dataclass
class ReportConfig(ExperimentConfig):
    pass


@dataclass
class SuiteConfig(ExperimentConfig):
    generate_data: bool = True
    run_pretrain: bool = True
    run_train: bool = True
    run_report: bool = True


@dataclass
class SweepConfig(SuiteConfig):
    sweep_name: str = "piroth_sweep"
    candidate_group: str = "passive_mm"