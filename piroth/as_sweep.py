from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .baselines import calibrate_avellaneda_stoikov, calibration_to_dict
from .config import DiagnosticsConfig
from .paper_evaluation import _fast_replay_policy, summarize_metrics
from .real_data import load_market_days
from .utils import save_json


def run_as_validation_sweep(
    config: DiagnosticsConfig,
    output_dir: Path,
    *,
    gammas: list[float] | None = None,
    horizons: list[int] | None = None,
    max_distance_ticks: list[int] | None = None,
    validation_days: int = 2,
) -> dict[str, object]:
    """Tune AS hyperparameters on train-heldout validation and evaluate best on test.

    The calibration/validation split is taken only from the configured training
    period. The configured test period is touched once for the selected
    validation winner.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    gammas = gammas or [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
    horizons = horizons or [16, 32, 64, 128, 256]
    max_distance_ticks = max_distance_ticks or [3, 6, 10, 15]

    train_days = load_market_days(config, "train", skip_msg=True)
    if len(train_days) <= validation_days:
        raise ValueError(f"Need more than {validation_days} training days for AS validation split.")
    calibration_days = train_days[:-validation_days]
    val_days = train_days[-validation_days:]
    test_days = load_market_days(config, "test", skip_msg=True)

    rows: list[dict[str, object]] = []
    calibration_payloads: dict[str, object] = {}
    for horizon in horizons:
        for max_distance in max_distance_ticks:
            tuned = copy.copy(config)
            tuned.as_fill_horizon_events = int(horizon)
            tuned.as_max_distance_ticks = int(max_distance)
            calibration = calibrate_avellaneda_stoikov(calibration_days, tuned)
            calibration_key = f"h{horizon}_d{max_distance}"
            calibration_payloads[calibration_key] = calibration_to_dict(calibration)
            for gamma in gammas:
                candidate = copy.copy(tuned)
                candidate.as_gamma = float(gamma)
                metrics = []
                for day in val_days:
                    metrics.extend(_fast_replay_policy("AS", day, candidate, calibration))
                summary = summarize_metrics(pd.DataFrame([asdict(metric) for metric in metrics])) if metrics else {}
                if not summary:
                    summary = {
                        "episodes": 0.0,
                        "pnl_mean": float("nan"),
                        "pnl_std": float("nan"),
                        "nd_pnl_mean": float("nan"),
                        "pnl_map_mean": float("nan"),
                        "profit_ratio_mean": float("nan"),
                        "avg_abs_position_mean": float("nan"),
                        "avg_spread_mean": float("nan"),
                        "turnover_mean": float("nan"),
                        "trades_mean": float("nan"),
                        "fill_rate_mean": float("nan"),
                        "reward_mean": float("nan"),
                        "sharpe": float("nan"),
                    }
                rows.append(
                    {
                        "symbol": config.symbol,
                        "data_source": config.data_source,
                        "split": "validation",
                        "as_gamma": float(gamma),
                        "as_fill_horizon_events": int(horizon),
                        "as_max_distance_ticks": int(max_distance),
                        **summary,
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "as_validation_sweep.csv", index=False)
    valid_frame = frame[frame["episodes"] > 0].copy() if "episodes" in frame else pd.DataFrame()
    if valid_frame.empty:
        raise ValueError("AS validation sweep produced no metrics.")
    best_row = valid_frame.sort_values(
        ["pnl_mean", "sharpe", "profit_ratio_mean"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()

    best_config = copy.copy(config)
    best_config.as_gamma = float(best_row["as_gamma"])
    best_config.as_fill_horizon_events = int(best_row["as_fill_horizon_events"])
    best_config.as_max_distance_ticks = int(best_row["as_max_distance_ticks"])
    best_calibration = calibrate_avellaneda_stoikov(calibration_days + val_days, best_config)
    test_metrics = []
    for day in test_days:
        test_metrics.extend(_fast_replay_policy("AS", day, best_config, best_calibration))
    test_frame = pd.DataFrame([asdict(metric) for metric in test_metrics])
    test_frame.to_csv(output_dir / "as_tuned_test_episodes.csv", index=False)
    test_summary = summarize_metrics(test_frame)

    summary = {
        "config": {
            "symbol": config.symbol,
            "data_source": config.data_source,
            "train_days": config.train_days,
            "test_days": config.test_days,
            "validation_days": validation_days,
            "episode_length": config.episode_length,
            "real_event_stride": config.real_event_stride,
            "events_per_day_override": config.events_per_day_override,
        },
        "search_space": {
            "as_gamma": gammas,
            "as_fill_horizon_events": horizons,
            "as_max_distance_ticks": max_distance_ticks,
        },
        "best_validation": best_row,
        "test_summary": test_summary,
        "test_calibration": calibration_to_dict(best_calibration),
        "calibrations": calibration_payloads,
    }
    save_json(output_dir / "as_validation_sweep_summary.json", summary)
    print(json.dumps({"best_validation": best_row, "test_summary": test_summary}, indent=2, sort_keys=True), flush=True)
    return summary
