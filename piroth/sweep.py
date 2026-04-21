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
                    "fair_value_signal_scale": 0.012,
                    "fair_value_noise_scale": 0.002,
                    "shock_event_prob": 0.006,
                    "shock_size_ticks": 3.5,
                    "metaorder_start_prob": 0.008,
                    "metaorder_persistence": 0.992,
                    "metaorder_scale": 1.8,
                    "market_order_impact_scale": 0.55,
                    "informed_taker_rate_scale": 0.8,
                    "noise_taker_rate_scale": 1.05,
                    "maker_add_rate_scale": 1.15,
                    "maker_cancel_rate_scale": 0.9,
                    "liquidity_refill_rate_scale": 1.1,
                    "maker_join_touch_prob_shift": 0.06,
                    "stress_liquidity_withdraw_scale": 0.45,
                    "market_order_alpha_sensitivity": 0.08,
                    "market_order_flow_sensitivity": 0.18,
                    "flow_reversion_scale": 0.65,
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
                    "fair_value_signal_scale": 0.012,
                    "fair_value_noise_scale": 0.002,
                    "shock_event_prob": 0.006,
                    "shock_size_ticks": 3.5,
                    "metaorder_start_prob": 0.008,
                    "metaorder_persistence": 0.992,
                    "metaorder_scale": 1.8,
                    "market_order_impact_scale": 0.55,
                    "informed_taker_rate_scale": 0.8,
                    "noise_taker_rate_scale": 1.05,
                    "maker_add_rate_scale": 1.15,
                    "maker_cancel_rate_scale": 0.9,
                    "liquidity_refill_rate_scale": 1.1,
                    "maker_join_touch_prob_shift": 0.06,
                    "stress_liquidity_withdraw_scale": 0.45,
                    "market_order_alpha_sensitivity": 0.08,
                    "market_order_flow_sensitivity": 0.18,
                    "flow_reversion_scale": 0.65,
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
                    "fair_value_signal_scale": 0.015,
                    "fair_value_noise_scale": 0.0025,
                    "shock_event_prob": 0.008,
                    "shock_size_ticks": 4.0,
                    "metaorder_start_prob": 0.01,
                    "metaorder_persistence": 0.994,
                    "metaorder_scale": 2.1,
                    "market_order_impact_scale": 0.45,
                    "informed_taker_rate_scale": 0.65,
                    "noise_taker_rate_scale": 1.1,
                    "maker_add_rate_scale": 1.25,
                    "maker_cancel_rate_scale": 0.82,
                    "liquidity_refill_rate_scale": 1.2,
                    "maker_join_touch_prob_shift": 0.1,
                    "stress_liquidity_withdraw_scale": 0.55,
                    "market_order_alpha_sensitivity": 0.06,
                    "market_order_flow_sensitivity": 0.14,
                    "flow_reversion_scale": 0.5,
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
