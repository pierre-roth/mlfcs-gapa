from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

import pyrallis

from .config import SuiteConfig, SweepConfig
from .run_suite import run_suite
from .utils import ensure_dir, save_json


def _suite_config(config: SweepConfig, name: str, overrides: dict[str, object]) -> SuiteConfig:
    payload = asdict(config)
    payload.update(overrides)
    payload["run_name"] = f"{config.run_name}_{name}"
    payload["data_dir"] = str(Path(config.data_dir) / name)
    allowed = {field.name for field in fields(SuiteConfig)}
    return SuiteConfig(**{key: value for key, value in payload.items() if key in allowed})


def _candidate_group(config: SweepConfig) -> list[tuple[str, dict[str, object]]]:
    if config.candidate_group == "passive_mm":
        return [
            (
                "passive_balanced",
                {
                    "mode": "medium",
                    "max_spread": 0.04,
                    "trade_reward_weight": 2.0,
                    "dampened_pnl_weight": 0.25,
                    "alpha_signal_scale": 0.75,
                    "market_order_impact_scale": 0.55,
                    "market_order_alpha_sensitivity": 0.08,
                    "market_order_flow_sensitivity": 0.18,
                    "flow_reversion_scale": 0.65,
                    "spread_widen_prob": 0.18,
                    "spread_imbalance_threshold": 0.45,
                    "spread_alpha_threshold": 0.28,
                    "recenter_follow_scale": 0.55,
                    "recenter_base_prob": 0.035,
                    "recenter_gap_scale": 0.08,
                    "recenter_alpha_scale": 0.02,
                    "market_order_tick_impact": 0.0009,
                    "market_order_alpha_impact": 0.00035,
                    "touch_replenish_fraction": 0.35,
                },
            ),
            (
                "passive_rebate",
                {
                    "mode": "medium",
                    "max_spread": 0.04,
                    "trade_reward_weight": 2.0,
                    "dampened_pnl_weight": 0.2,
                    "alpha_signal_scale": 0.75,
                    "market_order_impact_scale": 0.55,
                    "market_order_alpha_sensitivity": 0.08,
                    "market_order_flow_sensitivity": 0.18,
                    "flow_reversion_scale": 0.65,
                    "spread_widen_prob": 0.18,
                    "spread_imbalance_threshold": 0.45,
                    "spread_alpha_threshold": 0.28,
                    "recenter_follow_scale": 0.55,
                    "recenter_base_prob": 0.035,
                    "recenter_gap_scale": 0.08,
                    "recenter_alpha_scale": 0.02,
                    "market_order_tick_impact": 0.0009,
                    "market_order_alpha_impact": 0.00035,
                    "touch_replenish_fraction": 0.35,
                    "use_maker_rebate": True,
                    "maker_rebate_per_share": 0.0020,
                },
            ),
            (
                "passive_ultra_tight",
                {
                    "mode": "medium",
                    "max_spread": 0.03,
                    "trade_reward_weight": 2.5,
                    "dampened_pnl_weight": 0.15,
                    "alpha_signal_scale": 0.65,
                    "market_order_impact_scale": 0.45,
                    "market_order_alpha_sensitivity": 0.06,
                    "market_order_flow_sensitivity": 0.14,
                    "flow_reversion_scale": 0.5,
                    "spread_widen_prob": 0.10,
                    "spread_imbalance_threshold": 0.5,
                    "spread_alpha_threshold": 0.32,
                    "recenter_follow_scale": 0.45,
                    "recenter_base_prob": 0.03,
                    "recenter_gap_scale": 0.07,
                    "recenter_alpha_scale": 0.015,
                    "market_order_tick_impact": 0.0007,
                    "market_order_alpha_impact": 0.00025,
                    "touch_replenish_fraction": 0.55,
                },
            ),
        ]
    raise ValueError(f"Unknown candidate_group: {config.candidate_group}")


def _extract_metrics(summary: dict[str, object]) -> dict[str, float]:
    report = next(iter(summary.get("report", {}).values()), {}) if isinstance(summary.get("report"), dict) else {}
    pretrain = next(iter(summary.get("pretrain", {}).values()), {}) if isinstance(summary.get("pretrain"), dict) else {}
    return {
        "best_f1": float(pretrain.get("best_f1", 0.0)),
        "test_f1": float(pretrain.get("test_f1", 0.0)),
        "pnl_mean": float(report.get("pnl_mean", 0.0)),
        "sharpe": float(report.get("sharpe", 0.0)),
        "fill_rate_mean": float(report.get("fill_rate_mean", 0.0)),
        "trades_mean": float(report.get("trades_mean", 0.0)),
        "fixed1_pnl_mean": float(report.get("fixed1_pnl_mean", 0.0)),
    }


def run_sweep(config: SweepConfig) -> dict[str, object]:
    config.apply_mode_defaults()
    root = ensure_dir(Path(config.output_root) / config.sweep_name)
    summary: dict[str, object] = {}
    ranking = []
    for name, overrides in _candidate_group(config):
        candidate = _suite_config(config, name, overrides)
        result = run_suite(candidate)
        metrics = _extract_metrics(result)
        summary[name] = {"config": overrides, "metrics": metrics, "run_name": candidate.run_name}
        ranking.append(
            {
                "name": name,
                "score": (
                    metrics["fixed1_pnl_mean"] * 0.02
                    + metrics["fill_rate_mean"] * 500.0
                    + metrics["pnl_mean"] * 2.0
                    + metrics["sharpe"] * 5.0
                ),
                **metrics,
            }
        )
    ranking.sort(key=lambda item: item["score"], reverse=True)
    payload = {"candidates": summary, "ranking": ranking}
    save_json(root / "summary.json", payload)
    return payload


@pyrallis.wrap()
def main(config: SweepConfig) -> None:
    run_sweep(config)


if __name__ == "__main__":
    main()
