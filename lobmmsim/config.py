from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_dir: str = "data/simulated_processed"
    output_root: str = "artifacts_sim"
    mode: str = "full"
    symbols: list[str] = field(default_factory=lambda: ["000001", "000858", "002415"])
    session_windows: list[str] = field(default_factory=lambda: ["09:30:00-11:30:00", "13:00:00-15:00:00"])
    stable_windows: list[str] = field(default_factory=lambda: ["10:00:00-11:30:00", "13:00:00-14:30:00"])
    use_stable_hours: bool = True
    seed: int = 7
    run_name: str = ""
    device: str = "auto"

    num_days: int = 21
    train_days: int = 8
    val_days: int = 0
    test_days: int = 13
    latency: int = 1
    lookback: int = 50
    tick_size: float = 0.01
    trade_unit: int = 100
    max_inventory_units: int = 10
    max_rows_per_day: int | None = None
    max_pretrain_samples_per_day: int | None = None
    max_eval_episodes_per_day: int | None = None
    max_train_episodes_per_day: int | None = None

    pretrain_backbone: str = "attn"
    pretrain_horizon: int = 10
    pretrain_alpha: float = 1e-5
    pretrain_epochs: int = 10
    pretrain_batch_size: int = 128
    pretrain_num_workers: int = 0
    pretrain_lr: float = 1e-3
    pretrain_aux_task: str = "regime"
    pretrain_aux_weight: float = 0.35
    backbone_name: str = "attn_lob.pt"
    backbone_trainable: bool = True

    episode_length: int = 2000
    max_bias: float = 0.05
    max_spread: float = 0.10
    eta: float = 0.5
    zeta: float = 0.01
    zero_transaction_cost: bool = True
    deterministic_evaluation: bool = True
    eval_seed_base: int = 20260414

    rv_windows_s: list[int] = field(default_factory=lambda: [300, 600, 1800])
    rsi_windows_s: list[int] = field(default_factory=lambda: [300, 600, 1800])
    osi_windows_s: list[int] = field(default_factory=lambda: [10, 60, 300])

    ppo_epochs: int = 10
    ppo_rollouts_per_epoch: int = 8
    ppo_updates: int = 2
    ppo_minibatch_size: int = 128
    ppo_lr: float = 1e-4
    ppo_clip: float = 0.2
    ppo_checkpoint_every: int = 0
    ppo_select_best_model: bool = False
    ppo_selection_metric: str = "pnl_mean"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    gradient_clip_norm: float = 1.0
    bc_teacher: str = "fixed1"
    bc_epochs: int = 2
    bc_batch_size: int = 256
    bc_lr: float = 5e-4

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
    price_noise_scale: float = 0.0025
    report_top_attention_points: int = 50
    acceptance_seeds: list[int] = field(default_factory=lambda: [11, 19, 31, 37, 43])

    def apply_mode_defaults(self) -> "ExperimentConfig":
        if self.mode == "smoke":
            self.num_days = 4
            self.train_days = 2
            self.val_days = 0
            self.test_days = 2
            self.events_per_day = {symbol: min(self.events_per_day.get(symbol, 60_000), 4_000) for symbol in self.symbols}
            self.max_rows_per_day = self.max_rows_per_day or 4_000
            self.max_pretrain_samples_per_day = self.max_pretrain_samples_per_day or 1_024
            self.max_eval_episodes_per_day = self.max_eval_episodes_per_day or 1
            self.max_train_episodes_per_day = self.max_train_episodes_per_day or 1
            self.pretrain_epochs = min(self.pretrain_epochs, 1)
            self.pretrain_batch_size = min(self.pretrain_batch_size, 64)
            self.ppo_epochs = min(self.ppo_epochs, 1)
            self.ppo_rollouts_per_epoch = min(self.ppo_rollouts_per_epoch, 2)
            self.ppo_updates = min(self.ppo_updates, 1)
            self.ppo_minibatch_size = min(self.ppo_minibatch_size, 64)
            self.bc_epochs = min(self.bc_epochs, 1)
            self.bc_batch_size = min(self.bc_batch_size, 128)
        elif self.mode == "medium":
            self.num_days = min(self.num_days, 8)
            self.train_days = min(self.train_days, 4)
            self.val_days = min(self.val_days, 0)
            self.test_days = min(self.test_days, self.num_days - self.train_days)
            self.events_per_day = {symbol: min(self.events_per_day.get(symbol, 60_000), 20_000) for symbol in self.symbols}
            self.max_rows_per_day = self.max_rows_per_day or 20_000
            self.max_pretrain_samples_per_day = self.max_pretrain_samples_per_day or 10_000
            self.max_eval_episodes_per_day = self.max_eval_episodes_per_day or 2
            self.max_train_episodes_per_day = self.max_train_episodes_per_day or 4
            self.pretrain_epochs = min(self.pretrain_epochs, 3)
            self.ppo_epochs = min(self.ppo_epochs, 3)
            self.ppo_rollouts_per_epoch = min(self.ppo_rollouts_per_epoch, 4)
            self.bc_epochs = min(self.bc_epochs, 2)
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
        run_name = self.run_name or "sim_run"
        return Path(self.output_root) / run_name


@dataclass
class GenerateConfig(ExperimentConfig):
    overwrite: bool = False


@dataclass
class PretrainConfig(ExperimentConfig):
    pass


@dataclass
class RLTrainConfig(ExperimentConfig):
    reward_mode: str = "paper"

    def method_name(self) -> str:
        return "C_PPO"


@dataclass
class SuiteConfig(ExperimentConfig):
    generate_data: bool = True
    run_pretrain: bool = True
    run_rl: bool = True
    run_report: bool = True
    run_acceptance: bool = False
