from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd
import pyrallis

from .acceptance import run_acceptance_check
from .config import ExperimentConfig, GenerateConfig, PretrainConfig, RLTrainConfig, SuiteConfig
from .pretrain import run_pretrain
from .report import run_report
from .simulator import generate_dataset
from .train_rl import run_rl_training
from .utils import ensure_dir, save_json


@dataclass
class LearningMatrixConfig(SuiteConfig):
    matrix_name: str = "learning_matrix"
    include_acceptance: bool = True


def _cast_config(config, target_type, **overrides):
    source = asdict(config)
    source.update(overrides)
    allowed = {field.name for field in fields(target_type)}
    return target_type(**{key: value for key, value in source.items() if key in allowed})


def _apply_medium_matrix_defaults(config: LearningMatrixConfig) -> LearningMatrixConfig:
    config.apply_mode_defaults()
    config.symbols = config.symbols[:1]
    config.num_days = 8
    config.train_days = 4
    config.val_days = 1
    config.test_days = 3
    config.events_per_day = {symbol: min(config.events_per_day.get(symbol, 60_000), 12_000) for symbol in config.symbols}
    config.max_rows_per_day = min(config.max_rows_per_day or 12_000, 12_000)
    config.max_pretrain_samples_per_day = min(config.max_pretrain_samples_per_day or 6_000, 6_000)
    config.max_train_episodes_per_day = max(config.max_train_episodes_per_day or 0, 6)
    config.max_eval_episodes_per_day = max(config.max_eval_episodes_per_day or 3, 3)
    config.pretrain_epochs = max(config.pretrain_epochs, 4)
    config.pretrain_batch_size = min(config.pretrain_batch_size, 128)
    config.ppo_epochs = max(config.ppo_epochs, 4)
    config.ppo_rollouts_per_epoch = max(config.ppo_rollouts_per_epoch, 6)
    config.ppo_updates = max(config.ppo_updates, 2)
    config.ppo_minibatch_size = min(config.ppo_minibatch_size, 128)
    config.ppo_select_best_model = True
    config.bc_epochs = max(config.bc_epochs, 2)
    config.bc_batch_size = min(config.bc_batch_size, 256)
    config.device = "cpu"
    return config


def _variant_summary(
    variant: str,
    pretrain_summary: dict[str, dict[str, float | str]],
    ppo_summary: dict[str, dict[str, float]],
    report_summary: dict[str, dict[str, float]],
    symbol: str,
) -> dict[str, float | str]:
    pretrain = pretrain_summary[symbol]
    ppo = ppo_summary[symbol]
    report = report_summary[symbol]
    split_metrics = pretrain.get("split_metrics", {})
    test_metrics = split_metrics.get("test", {}) if isinstance(split_metrics, dict) else {}
    return {
        "variant": variant,
        "best_f1": float(pretrain.get("best_f1", 0.0)),
        "test_f1": float(test_metrics.get("f1", 0.0)) if isinstance(test_metrics, dict) else 0.0,
        "test_regime_accuracy": float(test_metrics.get("regime_accuracy", 0.0)) if isinstance(test_metrics, dict) else 0.0,
        "auxiliary_enabled": bool(pretrain.get("auxiliary_enabled", False)),
        "bc_samples": float(ppo.get("bc_samples", 0.0)),
        "bc_final_loss": float(ppo.get("bc_final_loss", 0.0)),
        "ppo_pnl_mean": float(report.get("pnl_mean", 0.0)),
        "ppo_sharpe": float(report.get("sharpe", 0.0)),
        "ppo_trades_mean": float(report.get("trades_mean", 0.0)),
        "ppo_reward_mean": float(report.get("reward_mean", 0.0)),
        "alpha_probe_r2": float(report.get("alpha_probe_r2", 0.0)),
        "regime_probe_accuracy": float(report.get("regime_probe_accuracy", 0.0)),
        "counterfactual_action1_shift": float(report.get("counterfactual_action1_shift", 0.0)),
        "fixed1_pnl_mean": float(report.get("fixed1_pnl_mean", 0.0)),
        "oracle_pnl_mean": float(report.get("oracle_pnl_mean", 0.0)),
    }


def run_learning_matrix(config: LearningMatrixConfig) -> dict[str, object]:
    config = _apply_medium_matrix_defaults(config)
    symbol = config.symbols[0]
    matrix_root = ensure_dir(Path(config.output_root) / config.matrix_name)
    data_dir = Path(config.data_dir)
    generate_dataset(_cast_config(config, GenerateConfig, overwrite=False))
    acceptance_summary = run_acceptance_check(_cast_config(config, ExperimentConfig), symbol=symbol) if config.include_acceptance else {}
    variants = [
        {
            "name": "scratch_aux",
            "pretrain_aux_task": "regime",
            "pretrain_aux_weight": config.pretrain_aux_weight,
            "bc_epochs": 0,
            "action_mode": "absolute",
            "reward_mode": "paper",
            "inventory_carry_penalty": config.inventory_carry_penalty,
        },
        {
            "name": "signed_mm_scratch_aux",
            "pretrain_aux_task": "regime",
            "pretrain_aux_weight": config.pretrain_aux_weight,
            "bc_epochs": 0,
            "action_mode": "signed_absolute",
            "reward_mode": "mm_only",
            "inventory_carry_penalty": max(config.inventory_carry_penalty, 0.01),
        },
        {
            "name": "signed_mm_bc_aux",
            "pretrain_aux_task": "regime",
            "pretrain_aux_weight": config.pretrain_aux_weight,
            "bc_epochs": config.bc_epochs,
            "action_mode": "signed_absolute",
            "reward_mode": "mm_only",
            "inventory_carry_penalty": max(config.inventory_carry_penalty, 0.01),
        },
    ]
    rows = []
    for variant in variants:
        run_name = f"{config.matrix_name}_{variant['name']}"
        variant_pretrain = run_pretrain(
            _cast_config(
                config,
                PretrainConfig,
                run_name=run_name,
                data_dir=str(data_dir),
                pretrain_aux_task=variant["pretrain_aux_task"],
                pretrain_aux_weight=variant["pretrain_aux_weight"],
                action_mode=variant["action_mode"],
            )
        )
        variant_rl = run_rl_training(
            _cast_config(
                config,
                RLTrainConfig,
                run_name=run_name,
                data_dir=str(data_dir),
                pretrain_aux_task=variant["pretrain_aux_task"],
                pretrain_aux_weight=variant["pretrain_aux_weight"],
                bc_epochs=variant["bc_epochs"],
                action_mode=variant["action_mode"],
                reward_mode=variant["reward_mode"],
                inventory_carry_penalty=variant["inventory_carry_penalty"],
            )
        )
        variant_report = run_report(
            _cast_config(
                config,
                RLTrainConfig,
                run_name=run_name,
                data_dir=str(data_dir),
                pretrain_aux_task=variant["pretrain_aux_task"],
                pretrain_aux_weight=variant["pretrain_aux_weight"],
                bc_epochs=variant["bc_epochs"],
                action_mode=variant["action_mode"],
                reward_mode=variant["reward_mode"],
                inventory_carry_penalty=variant["inventory_carry_penalty"],
            )
        )
        rows.append(_variant_summary(variant["name"], variant_pretrain, variant_rl, variant_report, symbol))
    frame = pd.DataFrame(rows)
    frame.to_csv(matrix_root / "summary.csv", index=False)
    summary = {
        "matrix_name": config.matrix_name,
        "symbol": symbol,
        "data_dir": str(data_dir),
        "acceptance": acceptance_summary,
        "variants": rows,
    }
    save_json(matrix_root / "summary.json", summary)
    return summary


@pyrallis.wrap()
def main(config: LearningMatrixConfig) -> None:
    run_learning_matrix(config)


if __name__ == "__main__":
    main()
