from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd
import pyrallis

from .acceptance import run_acceptance_check
from .config import ExperimentConfig
from .utils import ensure_dir, save_json


@dataclass
class CalibrationSweepConfig(ExperimentConfig):
    sweep_name: str = "synthetic_acceptance_sweep"


def _candidates() -> list[dict[str, object]]:
    return [
        {"name": "baseline"},
        {
            "name": "softer_impact",
            "alpha_signal_scale": 0.85,
            "market_order_impact_scale": 0.75,
            "recenter_follow_scale": 0.85,
        },
        {
            "name": "wider_quotes",
            "spread_widen_prob": 0.45,
            "spread_imbalance_threshold": 0.24,
            "spread_alpha_threshold": 0.15,
        },
        {
            "name": "mean_reverting_flow",
            "flow_reversion_scale": 1.35,
            "market_order_impact_scale": 0.8,
            "recenter_follow_scale": 0.85,
        },
        {
            "name": "balanced_passive",
            "alpha_signal_scale": 0.8,
            "market_order_impact_scale": 0.7,
            "flow_reversion_scale": 1.2,
            "spread_widen_prob": 0.42,
            "spread_imbalance_threshold": 0.24,
            "recenter_follow_scale": 0.8,
        },
        {
            "name": "low_adverse_selection",
            "alpha_signal_scale": 0.7,
            "price_noise_scale": 0.002,
            "market_order_impact_scale": 0.6,
            "flow_reversion_scale": 1.25,
            "spread_widen_prob": 0.40,
            "recenter_follow_scale": 0.75,
        },
    ]


def run_calibration_sweep(config: CalibrationSweepConfig) -> dict[str, object]:
    config.apply_mode_defaults()
    symbol = config.symbols[0]
    root = ensure_dir(Path(config.output_root) / config.sweep_name)
    experiment_fields = {field.name for field in fields(ExperimentConfig)}
    base_config = {key: value for key, value in asdict(config).items() if key in experiment_fields}
    rows: list[dict[str, object]] = []
    for candidate in _candidates():
        name = str(candidate["name"])
        overrides = {key: value for key, value in candidate.items() if key != "name"}
        candidate_cfg = ExperimentConfig(
            **{
                **base_config,
                **overrides,
                "symbols": [symbol],
                "run_name": f"{config.sweep_name}_{name}",
                "data_dir": str(root / name / "data"),
            }
        )
        candidate_cfg.apply_mode_defaults()
        summary = run_acceptance_check(candidate_cfg, symbol=symbol)
        rows.append(
            {
                "candidate": name,
                **overrides,
                **summary,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["fixed1_positive_seed_fraction", "fixed1_pnl_mean_avg", "oracle_paper_pnl_mean_avg"],
        ascending=[False, False, False],
    )
    frame.to_csv(root / "summary.csv", index=False)
    summary = {
        "symbol": symbol,
        "sweep_name": config.sweep_name,
        "candidates": frame.to_dict(orient="records"),
        "best_candidate": frame.iloc[0].to_dict() if not frame.empty else {},
    }
    save_json(root / "summary.json", summary)
    return summary


@pyrallis.wrap()
def main(config: CalibrationSweepConfig) -> None:
    run_calibration_sweep(config)


if __name__ == "__main__":
    main()
