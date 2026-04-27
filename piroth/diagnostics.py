from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import (
    AvellanedaStoikovPolicy,
    FixedLevelPolicy,
    calibration_to_dict,
    calibrate_avellaneda_stoikov,
    replay_policy,
    stable_episode_spans,
    summarize,
)
from .config import DiagnosticsConfig
from .data_quality import assess_synthetic_quality
from .plots import plot_episode_windows, plot_lob_heatmap, plot_lob_snapshots, plot_midprice_days, write_window_summary
from .paper_evaluation import evaluate_paper_baselines
from .real_data import load_market_days
from .simulator import SyntheticDay
from .utils import ensure_dir, save_json
from .visualizer import build_synthetic_data_report


def run_diagnostics(config: DiagnosticsConfig) -> dict[str, object]:
    config.apply_mode_defaults()
    output_dir = ensure_dir(config.output_dir())
    plots_dir = ensure_dir(output_dir / "plots")

    train_days = load_market_days(config, "train")
    test_days = load_market_days(config, "test")

    if config.export_generated_days:
        export_root = ensure_dir(config.export_dir())
        for synthetic_day in (train_days + test_days)[: config.export_day_count]:
            synthetic_day.export(export_root)

    calibration = calibrate_avellaneda_stoikov(train_days, config)
    fixed_policy = FixedLevelPolicy(config.fixed_level_baseline, config)
    as_policy = AvellanedaStoikovPolicy(calibration, config)

    fixed_results = [result for day in test_days for result in replay_policy(day, fixed_policy, config)]
    as_results = [result for day in test_days for result in replay_policy(day, as_policy, config)]

    fixed_frame = pd.DataFrame([asdict(result) for result in fixed_results])
    as_frame = pd.DataFrame([asdict(result) for result in as_results])
    fixed_frame.to_csv(output_dir / "fixed_baseline_episodes.csv", index=False)
    as_frame.to_csv(output_dir / "as_baseline_episodes.csv", index=False)

    market_summary = _market_summary(test_days, config)
    quality_summary = assess_synthetic_quality(test_days, config)
    paper_baselines = evaluate_paper_baselines(train_days, test_days, config, output_dir)
    sample_windows = _sample_windows(test_days, config)
    sample_day, start, stop = sample_windows[min(config.sample_episode_index, max(len(sample_windows) - 1, 0))]

    if config.create_plots:
        plot_midprice_days(test_days[: min(4, len(test_days))], plots_dir / "midprice_daily_overview.png")
        plot_episode_windows(test_days, sample_windows, plots_dir / "midprice_2000_event_windows.png")
        plot_lob_heatmap(sample_day, start, stop, plots_dir / "lob_heatmap_episode.png")
        plot_lob_snapshots(sample_day, [start, start + (stop - start) // 2, stop - 1], plots_dir / "lob_snapshots_episode.png")
        write_window_summary(sample_windows, plots_dir / "window_summary.csv")
        report_path = build_synthetic_data_report(test_days[: min(len(test_days), 4)], config, output_dir / "visual_report")
    else:
        report_path = output_dir / "visual_report" / "index.html"

    summary = {
        "config": asdict(config),
        "market_summary": market_summary,
        "synthetic_quality": quality_summary,
        "as_calibration": calibration_to_dict(calibration),
        "fixed_baseline": summarize(fixed_results),
        "as_baseline": summarize(as_results),
        "paper_baselines": paper_baselines,
        "visual_report": str(report_path),
        "sample_episode": {
            "day": sample_day.day,
            "start_event": start,
            "stop_event": stop,
        },
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def _market_summary(days: list[SyntheticDay], config: DiagnosticsConfig) -> dict[str, object]:
    rows = []
    window_moves = []
    for day in days:
        price = day.price
        rows.append(
            {
                "day": day.day,
                "events": int(len(price)),
                "midprice_std_bp": float(price["return_bp"].std(ddof=0)),
                "spread_mean_ticks": float(np.mean(price["spread_ticks"])),
                "spread_p95_ticks": float(np.quantile(price["spread_ticks"], 0.95)),
                "absolute_return_bp_mean": float(np.mean(np.abs(price["return_bp"]))),
            }
        )
        for start, stop in stable_episode_spans(day, config):
            segment = price.iloc[start:stop]
            start_mid = float(segment.iloc[0]["midprice"])
            end_mid = float(segment.iloc[-1]["midprice"])
            window_moves.append(
                {
                    "day": day.day,
                    "start_event": start,
                    "stop_event": stop,
                    "move_ticks": float((end_mid - start_mid) / config.symbol_spec.tick_size),
                    "return_bp": float(10_000.0 * (end_mid / start_mid - 1.0)),
                }
            )
    market_frame = pd.DataFrame(rows)
    window_frame = pd.DataFrame(window_moves)
    return {
        "per_day": rows,
        "window_move_abs_ticks_mean": float(np.mean(np.abs(window_frame["move_ticks"]))) if not window_frame.empty else 0.0,
        "window_move_abs_ticks_p50": float(np.median(np.abs(window_frame["move_ticks"]))) if not window_frame.empty else 0.0,
        "window_move_abs_ticks_p90": float(np.quantile(np.abs(window_frame["move_ticks"]), 0.90)) if not window_frame.empty else 0.0,
        "spread_mean_ticks": float(market_frame["spread_mean_ticks"].mean()) if not market_frame.empty else 0.0,
        "events_per_day_mean": float(market_frame["events"].mean()) if not market_frame.empty else 0.0,
    }


def _sample_windows(days: list[SyntheticDay], config: DiagnosticsConfig) -> list[tuple[SyntheticDay, int, int]]:
    candidates: list[tuple[SyntheticDay, int, int]] = []
    for day in days:
        for span in stable_episode_spans(day, config):
            candidates.append((day, *span))
    if not candidates:
        raise RuntimeError("No stable 2000-event episodes were generated")
    if len(candidates) <= config.random_window_count:
        return candidates
    rng = np.random.default_rng(config.seed)
    idx = np.sort(rng.choice(len(candidates), size=config.random_window_count, replace=False))
    return [candidates[int(i)] for i in idx]
