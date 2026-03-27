from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_dir: str = "data/processed"
    output_root: str = "artifacts"
    mode: str = "smoke"
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "GOOGL"])
    session_start: str = "09:30:00"
    session_end: str = "16:00:00"
    use_stable_hours: bool = True
    stable_windows: list[str] = field(default_factory=lambda: ["10:00:00-15:30:00"])
    lookback: int = 50
    tick_size: float = 0.01
    device: str = "auto"
    seed: int = 7
    run_name: str = ""
    train_days: int = 7
    val_days: int = 1
    test_days: int = 2
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
    pretrain_prefetch_factor: int | None = None
    pretrain_balance_mode: str = "weighted_loss"
    pretrain_sampler_power: float = 1.0
    pretrain_eval_samples_per_day: int | None = None
    pretrain_lr: float = 1e-3
    pretrain_checkpoint_seconds: int = 600
    pretrain_resume: bool = True
    episode_length: int = 2000
    target_episode_seconds: int | None = 120
    latency: int = 1
    max_inventory: int = 200
    trade_unit: int = 1
    max_bias: float = 0.05
    max_spread: float = 0.10
    quote_scale_mode: str = "bps"
    max_bias_bps: float = 2.0
    max_spread_bps: float = 8.0
    eta: float = 0.5
    zeta: float = 0.01
    zeta_start: float | None = None
    zeta_end: float | None = None
    zeta_l1: float = 0.0
    zeta_l2: float | None = None
    gamma: float = 0.99
    normalize_advantages: bool = True
    gradient_clip_norm: float = 1.0
    deterministic_evaluation: bool = True
    eval_seed_base: int = 20260326
    ppo_epochs: int = 10
    ppo_rollouts_per_epoch: int = 16
    ppo_updates: int = 2
    ppo_minibatch_size: int = 128
    ppo_lr: float = 3e-4
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    dqn_epochs: int = 10
    dqn_batches_per_epoch: int = 64
    dqn_batch_size: int = 128
    dqn_lr: float = 1e-4
    dqn_replay_size: int = 20_000
    dqn_warmup_steps: int = 1_000
    dqn_target_interval: int = 250
    dqn_eps_start: float = 1.0
    dqn_eps_end: float = 0.05
    dqn_eps_decay: int = 5_000
    latency_sweep: list[int] = field(default_factory=lambda: [1, 2, 5, 10])
    as_gamma: float = 0.1
    as_kappa: float = 20.0
    as_vol_window_s: int = 300
    report_top_attention_points: int = 50

    def apply_mode_defaults(self) -> "ExperimentConfig":
        if self.mode == "smoke":
            self.train_days = min(self.train_days, 2)
            self.val_days = min(self.val_days, 1)
            self.test_days = min(self.test_days, 1)
            self.max_rows_per_day = self.max_rows_per_day or 15_000
            self.max_pretrain_samples_per_day = self.max_pretrain_samples_per_day or 2_048
            self.max_eval_episodes_per_day = self.max_eval_episodes_per_day or 1
            self.max_train_episodes_per_day = self.max_train_episodes_per_day or 1
            self.pretrain_epochs = min(self.pretrain_epochs, 2)
            self.pretrain_batch_size = min(self.pretrain_batch_size, 64)
            if self.pretrain_eval_samples_per_day is None:
                self.pretrain_eval_samples_per_day = 1_024
            self.ppo_epochs = min(self.ppo_epochs, 2)
            self.ppo_rollouts_per_epoch = min(self.ppo_rollouts_per_epoch, 2)
            self.ppo_updates = min(self.ppo_updates, 1)
            self.ppo_minibatch_size = min(self.ppo_minibatch_size, 64)
            self.dqn_epochs = min(self.dqn_epochs, 2)
            self.dqn_batches_per_epoch = min(self.dqn_batches_per_epoch, 2)
            self.dqn_batch_size = min(self.dqn_batch_size, 64)
            self.dqn_replay_size = min(self.dqn_replay_size, 2_000)
            self.dqn_warmup_steps = min(self.dqn_warmup_steps, 100)
        elif self.mode == "medium":
            self.train_days = min(self.train_days, 4)
            self.val_days = min(self.val_days, 1)
            self.test_days = min(self.test_days, 1)
            self.max_rows_per_day = self.max_rows_per_day or 200_000
            self.max_pretrain_samples_per_day = self.max_pretrain_samples_per_day or 50_000
            self.max_eval_episodes_per_day = self.max_eval_episodes_per_day or 4
            self.max_train_episodes_per_day = self.max_train_episodes_per_day or 8
            self.pretrain_epochs = min(self.pretrain_epochs, 4)
            if self.pretrain_batch_size == 128:
                self.pretrain_batch_size = 256
            if self.pretrain_eval_samples_per_day is None:
                self.pretrain_eval_samples_per_day = 12_500
            self.ppo_epochs = min(self.ppo_epochs, 4)
            self.ppo_rollouts_per_epoch = min(self.ppo_rollouts_per_epoch, 4)
            self.ppo_updates = min(self.ppo_updates, 1)
            if self.ppo_minibatch_size == 128:
                self.ppo_minibatch_size = 256
            self.dqn_epochs = min(self.dqn_epochs, 4)
            self.dqn_batches_per_epoch = min(self.dqn_batches_per_epoch, 16)
            if self.dqn_batch_size == 128:
                self.dqn_batch_size = 256
            self.dqn_replay_size = min(self.dqn_replay_size, 10_000)
            self.dqn_warmup_steps = min(self.dqn_warmup_steps, 500)
        else:
            self.max_rows_per_day = self.max_rows_per_day
            self.max_pretrain_samples_per_day = self.max_pretrain_samples_per_day
            self.max_eval_episodes_per_day = self.max_eval_episodes_per_day
            self.max_train_episodes_per_day = self.max_train_episodes_per_day
        if self.pretrain_eval_samples_per_day is None:
            base = self.max_pretrain_samples_per_day or 100_000
            self.pretrain_eval_samples_per_day = max(4_096, base // 4)
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
        run_name = self.run_name or f"{self.mode}_run"
        return Path(self.output_root) / run_name


@dataclass
class PretrainConfig(ExperimentConfig):
    save_backbone_name: str = "attn_lob.pt"


@dataclass
class RLTrainConfig(ExperimentConfig):
    algorithm: str = "ppo"
    state_mode: str = "full"
    reward_mode: str = "hybrid"
    backbone_name: str = "attn_lob.pt"
    backbone_run_name: str = ""
    backbone_trainable: bool = True
    wo_lob_state: bool = False
    wo_dynamic_state: bool = False
    alt_backbone: str = "simple"
    variant_tag: str = ""

    def variant_name(self) -> str:
        if self.state_mode == "inventory_only":
            return "inventory_only"
        if self.state_mode == "handcrafted":
            return "handcrafted"
        parts = ["full"]
        if self.wo_lob_state:
            parts.append("wo_lob")
        if self.wo_dynamic_state:
            parts.append("wo_dynamic")
        if self.variant_tag == "wo_attn":
            parts.append("simple_backbone")
        return "_".join(parts)

    def method_name(self) -> str:
        if self.state_mode == "inventory_only":
            return "PPO_inventory_only"
        if self.state_mode == "handcrafted":
            return "PPO_handcrafted"
        if self.wo_lob_state:
            return "PPO_wo_lob"
        if self.wo_dynamic_state:
            return "PPO_wo_dynamic"
        if self.variant_tag == "wo_attn":
            return "PPO_wo_attn"
        return "PPO_full"


@dataclass
class SuiteConfig(ExperimentConfig):
    run_pretrain: bool = True
    run_main_agents: bool = True
    run_rl_baselines: bool = False
    run_non_rl_baselines: bool = True
    run_ablations: bool = True
    run_latency: bool = True
    run_report: bool = True
