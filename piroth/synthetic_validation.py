from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .baselines import calibrate_avellaneda_stoikov
from .config import DEFAULT_SYMBOLS, DiagnosticsConfig
from .data_quality import assess_synthetic_quality
from .simulator import SyntheticMarketGenerator
from .utils import ensure_dir, save_json
from .visualizer import build_synthetic_data_report


DEFAULT_VALIDATION_SEEDS = [7, 11, 17, 23]
DEFAULT_VALIDATION_SYMBOLS = ["000001", "000858", "002415"]


def run_synthetic_validation_suite(
    config: DiagnosticsConfig,
    *,
    seeds: list[int] | None = None,
    symbols: list[str] | None = None,
    days_per_case: int = 2,
    events_per_day: int | None = 12_000,
    report_cases: int = 4,
    export_report_days: bool = True,
) -> dict[str, object]:
    config.apply_mode_defaults()
    seeds = seeds or DEFAULT_VALIDATION_SEEDS
    symbols = symbols or DEFAULT_VALIDATION_SYMBOLS
    output_dir = ensure_dir(config.output_dir())
    cases_dir = ensure_dir(output_dir / "cases")

    rows = []
    for symbol in symbols:
        if symbol not in DEFAULT_SYMBOLS:
            raise KeyError(f"Unsupported validation symbol {symbol!r}")
        for seed in seeds:
            case_id = f"{symbol}_seed{seed}"
            case_config = _case_config(config, symbol, seed, days_per_case, events_per_day, case_id)
            generator = SyntheticMarketGenerator(case_config)
            days = [generator.generate_day(day) for day in generator.test_days()]
            quality = assess_synthetic_quality(days, case_config)
            calibration = calibrate_avellaneda_stoikov(days, case_config)
            row = _case_row(case_id, case_config, quality, calibration.fill_profile)
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["passes_quality_gate", "score", "symbol", "seed"], ascending=[True, True, True, True])
    frame.to_csv(output_dir / "synthetic_validation_cases.csv", index=False)
    by_symbol = _by_symbol(frame)
    by_symbol.to_csv(output_dir / "synthetic_validation_by_symbol.csv", index=False)

    selected = _selected_report_cases(frame, report_cases)
    report_paths = []
    for row in selected.itertuples(index=False):
        case_id = str(row.case_id)
        case_config = _case_config(config, str(row.symbol), int(row.seed), days_per_case, events_per_day, case_id)
        generator = SyntheticMarketGenerator(case_config)
        days = [generator.generate_day(day) for day in generator.test_days()]
        case_root = ensure_dir(cases_dir / case_id)
        report_path = build_synthetic_data_report(days, case_config, case_root / "visual_report")
        report_paths.append({"case_id": case_id, "visual_report": str(report_path)})
        if export_report_days:
            for day in days[: case_config.export_day_count]:
                day.export(case_root / "exported_days")

    summary = {
        "config": asdict(config),
        "seeds": seeds,
        "symbols": symbols,
        "days_per_case": days_per_case,
        "events_per_day": events_per_day,
        "case_count": int(len(frame)),
        "pass_rate": float(frame["passes_quality_gate"].mean()) if not frame.empty else 0.0,
        "score_mean": float(frame["score"].mean()) if not frame.empty else 0.0,
        "score_min": float(frame["score"].min()) if not frame.empty else 0.0,
        "flags_total": int(frame["flag_count"].sum()) if not frame.empty else 0,
        "worst_cases": frame.head(min(10, len(frame))).to_dict(orient="records"),
        "by_symbol": by_symbol.to_dict(orient="records"),
        "reports": report_paths,
    }
    save_json(output_dir / "synthetic_validation_summary.json", summary)
    _write_index(output_dir, frame, by_symbol, report_paths, summary)
    return summary


def _case_config(
    config: DiagnosticsConfig,
    symbol: str,
    seed: int,
    days_per_case: int,
    events_per_day: int | None,
    case_id: str,
) -> DiagnosticsConfig:
    return replace(
        config,
        symbol=symbol,
        seed=seed,
        run_name=f"{config.run_name}_{case_id}",
        num_days=days_per_case,
        train_days=0,
        test_days=days_per_case,
        events_per_day_override=events_per_day,
        export_day_count=min(config.export_day_count, days_per_case),
        create_plots=False,
        export_generated_days=False,
    )


def _case_row(case_id: str, config: DiagnosticsConfig, quality: dict[str, object], fill_profile: dict[int, float]) -> dict[str, object]:
    flags = [str(flag) for flag in quality.get("flags", [])]
    fill_values = {f"fill_prob_{distance}": float(fill_profile.get(distance, 0.0)) for distance in range(config.as_max_distance_ticks + 1)}
    fill_curve_ok = _fill_curve_ok(fill_profile)
    gate_reasons = _gate_reasons(quality, flags, fill_curve_ok)
    passes_quality_gate = not gate_reasons
    return {
        "case_id": case_id,
        "symbol": config.symbol,
        "seed": config.seed,
        "events_per_day": config.events_per_day_override or config.symbol_spec.events_per_day,
        "days": config.test_days,
        "passes_quality_gate": passes_quality_gate,
        "gate_reasons": "; ".join(gate_reasons),
        "flag_count": len(flags),
        "flags": "; ".join(flags),
        "fill_curve_ok": fill_curve_ok,
        **{key: quality[key] for key in quality if key != "flags"},
        **fill_values,
    }


def _gate_reasons(quality: dict[str, object], flags: list[str], fill_curve_ok: bool) -> list[str]:
    reasons = list(flags)
    if float(quality["score"]) < 90.0:
        reasons.append("quality score below 90")
    if not 1.2 <= float(quality["spread_mean_ticks"]) <= 3.0:
        reasons.append("mean spread outside 1.2-3.0 ticks")
    if float(quality["spread_p95_ticks"]) > 8.0:
        reasons.append("p95 spread above 8 ticks")
    if not 2.0 <= float(quality["window_abs_move_p50_ticks"]) <= 30.0:
        reasons.append("median 2000-event move outside 2-30 ticks")
    if not 10.0 <= float(quality["window_abs_move_p90_ticks"]) <= 75.0:
        reasons.append("p90 2000-event move outside 10-75 ticks")
    if not 0.25 <= float(quality["trades_per_event"]) <= 1.20:
        reasons.append("trade density outside 0.25-1.20 trades/event")
    if float(quality["order_flow_autocorr"]) < 0.05:
        reasons.append("order-flow autocorrelation below 0.05")
    if not fill_curve_ok:
        reasons.append("fill curve does not decay plausibly")
    return reasons


def _fill_curve_ok(fill_profile: dict[int, float]) -> bool:
    if not fill_profile:
        return False
    ordered = [float(fill_profile.get(distance, 0.0)) for distance in sorted(fill_profile)]
    if ordered[0] < 0.15:
        return False
    return all(left + 0.03 >= right for left, right in zip(ordered, ordered[1:], strict=False))


def _by_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    columns = [
        "passes_quality_gate",
        "score",
        "trades_per_event",
        "spread_mean_ticks",
        "spread_p95_ticks",
        "window_abs_move_p50_ticks",
        "window_abs_move_p90_ticks",
        "order_flow_autocorr",
        "depth_imbalance_std",
    ]
    grouped = frame.groupby("symbol")[columns].agg(["mean", "min", "max"])
    grouped.columns = ["_".join(column).strip("_") for column in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def _selected_report_cases(frame: pd.DataFrame, report_cases: int) -> pd.DataFrame:
    if report_cases <= 0 or frame.empty:
        return frame.head(0)
    worst = frame.sort_values(["passes_quality_gate", "score"], ascending=[True, True]).head(report_cases)
    if len(worst) >= report_cases:
        return worst
    best = frame.sort_values("score", ascending=False).head(report_cases - len(worst))
    return pd.concat([worst, best], ignore_index=True).drop_duplicates("case_id").head(report_cases)


def _write_index(
    output_dir: Path,
    frame: pd.DataFrame,
    by_symbol: pd.DataFrame,
    report_paths: list[dict[str, str]],
    summary: dict[str, object],
) -> None:
    report_links = {
        item["case_id"]: Path(item["visual_report"]).relative_to(output_dir).as_posix()
        for item in report_paths
        if Path(item["visual_report"]).is_relative_to(output_dir)
    }
    table = frame.copy()
    table["visual_report"] = table["case_id"].map(lambda case_id: _link(report_links.get(str(case_id), ""), "open"))
    columns = [
        "case_id",
        "visual_report",
        "passes_quality_gate",
        "gate_reasons",
        "score",
        "flags",
        "symbol",
        "seed",
        "trades_per_event",
        "spread_mean_ticks",
        "spread_p95_ticks",
        "window_abs_move_p50_ticks",
        "window_abs_move_p90_ticks",
        "order_flow_autocorr",
        "depth_imbalance_std",
        "fill_curve_ok",
        "fill_prob_0",
        "fill_prob_1",
        "fill_prob_2",
    ]
    available = [column for column in columns if column in table.columns]
    html = "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>Synthetic Validation</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:0;background:#f8fafc;color:#0f172a}",
            "header{padding:26px 34px;background:#111827;color:white}",
            "main{max-width:1380px;margin:0 auto;padding:24px}",
            "table{border-collapse:collapse;width:100%;font-size:13px;background:white;border:1px solid #e2e8f0;margin:14px 0 28px}",
            "th,td{padding:7px 9px;border-bottom:1px solid #e2e8f0;text-align:right;vertical-align:top}",
            "th:first-child,td:first-child{text-align:left}",
            "td:nth-child(2),th:nth-child(2){text-align:left}",
            ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:16px 0 24px}",
            ".card{background:white;border:1px solid #e2e8f0;border-radius:8px;padding:13px}",
            ".card b{display:block;color:#475569;font-size:12px;text-transform:uppercase}.card span{font-size:22px;font-weight:650}",
            "a{color:#2563eb;text-decoration:none}",
            "</style></head><body>",
            "<header><h1>Synthetic Data Validation</h1><div>Multi-seed quality gate for paper-replication LOB data</div></header>",
            "<main>",
            "<div class='cards'>",
            _card("Cases", summary["case_count"]),
            _card("Pass Rate", f"{100.0 * float(summary['pass_rate']):.1f}%"),
            _card("Mean Score", f"{float(summary['score_mean']):.2f}"),
            _card("Min Score", f"{float(summary['score_min']):.2f}"),
            _card("Total Flags", summary["flags_total"]),
            "</div>",
            "<h2>Cases</h2>",
            table[available].to_html(index=False, escape=False, float_format=lambda value: f"{value:.4f}"),
            "<h2>By Symbol</h2>",
            by_symbol.to_html(index=False, escape=False, float_format=lambda value: f"{value:.4f}") if not by_symbol.empty else "<p>No symbol summary.</p>",
            "</main></body></html>",
        ]
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _card(name: str, value: object) -> str:
    return f"<div class='card'><b>{name}</b><span>{value}</span></div>"


def _link(path: str, label: str) -> str:
    if not path:
        return ""
    return f"<a href='{path}'>{label}</a>"
