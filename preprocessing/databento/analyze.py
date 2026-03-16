from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .common import csv_reader, discover_schema_jobs_for_symbol, int_to_price, parse_session_time
from .convert import collect_schema_files
from .progress import DatabentoProgress
from .validate import (
    LEVELS,
    Snapshot,
    build_processed_sample_set,
    local_timestamp_to_utc_event,
    normalize_book_side,
    snapshot_bbo,
    utc_session_bounds,
)


@dataclass
class DivergenceConfig:
    raw_root: Path
    processed_root: Path
    analysis_root: Path
    symbol: str
    timezone_name: str = "America/New_York"
    session_start: str = "09:30:00"
    session_end: str = "16:00:00"
    sample_every: int = 10_000
    edge_samples: int = 1_000
    tolerance_us: int = 1_000
    examples: int = 10
    days: Optional[Set[str]] = None


@dataclass
class ExampleDiff:
    timestamp: str
    match_mode: str
    matched_timestamp: Optional[str]
    delta_us: Optional[float]
    category: str
    processed_bbo: Dict[str, str]
    matched_bbo: Optional[Dict[str, str]]
    changed_levels: List[str]


@dataclass
class DayDivergenceResult:
    day: str
    sample_timestamps: int
    exact_timestamp_hits: int
    nearby_hits_within_tolerance: int
    missing_without_nearby_match: int
    exact_top10_matches: int
    exact_bbo_matches: int
    nearby_top10_matches: int
    nearby_bbo_matches: int
    category_counts: Dict[str, int]
    price_level_mismatch_counts: Dict[str, int]
    size_level_mismatch_counts: Dict[str, int]
    example_diffs: List[ExampleDiff] = field(default_factory=list)


@dataclass
class CandidateMatch:
    raw_ts: str
    snapshot: Snapshot


@dataclass
class SampleMatchRecord:
    local_ts: str
    raw_ts: str
    processed_snapshot: Snapshot
    exact: Optional[CandidateMatch] = None
    prev_candidate: Optional[CandidateMatch] = None
    next_candidate: Optional[CandidateMatch] = None


def raw_timestamp_to_ns(timestamp: str) -> int:
    date_part, time_part = timestamp[:-1].split("T")
    y, m, d = map(int, date_part.split("-"))
    hms_part, fractional = (time_part.split(".", 1) + [""])[:2]
    hh, mm, ss = map(int, hms_part.split(":"))
    fractional = (fractional + "000000000")[:9]
    # This is only used for same-day relative comparisons, so lexicographic day assumptions hold.
    return (((((y * 100 + m) * 100 + d) * 24 + hh) * 60 + mm) * 60 + ss) * 1_000_000_000 + int(fractional)


def bbo_to_dict(snapshot: Snapshot) -> Dict[str, str]:
    (ask_price, ask_size), (bid_price, bid_size) = snapshot_bbo(snapshot)
    return {
        "ask_price": int_to_price(ask_price),
        "ask_size": str(ask_size),
        "bid_price": int_to_price(bid_price),
        "bid_size": str(bid_size),
    }


def analyze_snapshot_difference(processed: Snapshot, reference: Snapshot) -> tuple[str, List[str], Dict[str, int], Dict[str, int]]:
    price_counts: Dict[str, int] = {}
    size_counts: Dict[str, int] = {}
    changed_levels: List[str] = []
    any_price_diff = False
    any_size_diff = False

    for side_name, processed_levels, reference_levels in (
        ("ask", processed[0], reference[0]),
        ("bid", processed[1], reference[1]),
    ):
        for idx, (processed_level, reference_level) in enumerate(zip(processed_levels, reference_levels), start=1):
            processed_price, processed_size = processed_level
            reference_price, reference_size = reference_level
            level_key = f"{side_name}{idx}"
            if processed_price != reference_price:
                any_price_diff = True
                price_counts[level_key] = price_counts.get(level_key, 0) + 1
                changed_levels.append(f"{level_key}:price")
            elif processed_size != reference_size:
                any_size_diff = True
                size_counts[level_key] = size_counts.get(level_key, 0) + 1
                changed_levels.append(f"{level_key}:size")

    if not any_price_diff and not any_size_diff:
        category = "exact_top10_match"
    elif not any_price_diff:
        category = "size_only_difference"
    elif snapshot_bbo(processed) == snapshot_bbo(reference):
        category = "deeper_book_price_difference"
    else:
        category = "bbo_price_difference"

    return category, changed_levels[:12], price_counts, size_counts


def choose_nearest_candidate(record: SampleMatchRecord, tolerance_ns: int) -> tuple[Optional[str], Optional[CandidateMatch], Optional[float]]:
    processed_ns = raw_timestamp_to_ns(record.raw_ts)
    candidates: List[tuple[int, str, CandidateMatch]] = []
    if record.prev_candidate is not None:
        prev_delta = abs(processed_ns - raw_timestamp_to_ns(record.prev_candidate.raw_ts))
        candidates.append((prev_delta, "nearest_prev", record.prev_candidate))
    if record.next_candidate is not None:
        next_delta = abs(raw_timestamp_to_ns(record.next_candidate.raw_ts) - processed_ns)
        candidates.append((next_delta, "nearest_next", record.next_candidate))
    if not candidates:
        return None, None, None
    delta_ns, mode, candidate = min(candidates, key=lambda item: item[0])
    if delta_ns > tolerance_ns:
        return None, None, delta_ns / 1_000
    return mode, candidate, delta_ns / 1_000


def build_sample_records(processed_day_dir: Path, sample_every: int, edge_samples: int, timezone_name: str) -> List[SampleMatchRecord]:
    processed_samples = build_processed_sample_set(
        processed_day_dir,
        sample_every=sample_every,
        edge_samples=edge_samples,
    )
    records: List[SampleMatchRecord] = []
    for local_ts, snapshot in processed_samples.samples.items():
        records.append(
            SampleMatchRecord(
                local_ts=local_ts,
                raw_ts=local_timestamp_to_utc_event(local_ts, timezone_name),
                processed_snapshot=snapshot,
            )
        )
    return records


def scan_mbp_for_records(
    path: Path,
    day: str,
    timezone_name: str,
    session_start: str,
    session_end: str,
    records: List[SampleMatchRecord],
    progress: Optional[DatabentoProgress] = None,
) -> None:
    session_start_time = parse_session_time(session_start)
    session_end_time = parse_session_time(session_end)
    start_prefix, end_prefix = utc_session_bounds(day, timezone_name, session_start_time, session_end_time)
    progress_task = None
    if progress is not None:
        progress_task = progress.add_task(f"[magenta]{day}[/magenta] MBP-10 analysis", total=None)

    sample_index = 0
    current_ts: Optional[str] = None
    current_snapshot: Optional[Snapshot] = None
    previous_candidate: Optional[CandidateMatch] = None

    def finalize_group(raw_ts: Optional[str], snapshot: Optional[Snapshot]) -> None:
        nonlocal sample_index, previous_candidate
        if raw_ts is None or snapshot is None:
            return
        candidate = CandidateMatch(raw_ts=raw_ts, snapshot=snapshot)
        while sample_index < len(records) and records[sample_index].raw_ts < raw_ts:
            record = records[sample_index]
            record.prev_candidate = previous_candidate
            record.next_candidate = candidate
            sample_index += 1
        while sample_index < len(records) and records[sample_index].raw_ts == raw_ts:
            record = records[sample_index]
            record.exact = candidate
            record.prev_candidate = previous_candidate
            record.next_candidate = candidate
            sample_index += 1
        previous_candidate = candidate

    for idx, row in enumerate(csv_reader(path), start=1):
        raw_ts = row["ts_event"]
        raw_prefix = raw_ts[:19]
        if raw_prefix > end_prefix:
            break
        if raw_prefix < start_prefix:
            if progress is not None and progress_task is not None and idx % 100_000 == 0:
                progress.update(progress_task, description=f"[magenta]{day}[/magenta] MBP-10 rows: {idx:,}")
            continue

        bids = normalize_book_side(row, "bid")
        asks = normalize_book_side(row, "ask")
        if not bids or not asks:
            if progress is not None and progress_task is not None and idx % 100_000 == 0:
                progress.update(progress_task, description=f"[magenta]{day}[/magenta] MBP-10 rows: {idx:,}")
            continue

        snapshot = (asks, bids)
        if current_ts is None:
            current_ts = raw_ts
            current_snapshot = snapshot
        elif raw_ts == current_ts:
            current_snapshot = snapshot
        else:
            finalize_group(current_ts, current_snapshot)
            current_ts = raw_ts
            current_snapshot = snapshot

        if progress is not None and progress_task is not None and idx % 100_000 == 0:
            progress.update(progress_task, description=f"[magenta]{day}[/magenta] MBP-10 rows: {idx:,}")

    finalize_group(current_ts, current_snapshot)
    while sample_index < len(records):
        records[sample_index].prev_candidate = previous_candidate
        sample_index += 1

    if progress is not None and progress_task is not None:
        progress.update(progress_task, description=f"[magenta]{day}[/magenta] MBP-10 analysis complete", completed=1, total=1)
        progress.remove_task(progress_task)


def analyze_day(
    symbol: str,
    day: str,
    processed_root: Path,
    mbp_path: Path,
    timezone_name: str,
    session_start: str,
    session_end: str,
    sample_every: int,
    edge_samples: int,
    tolerance_us: int,
    examples: int,
    progress: Optional[DatabentoProgress] = None,
) -> DayDivergenceResult:
    records = build_sample_records(
        processed_day_dir=processed_root / symbol / day,
        sample_every=sample_every,
        edge_samples=edge_samples,
        timezone_name=timezone_name,
    )
    scan_mbp_for_records(
        path=mbp_path,
        day=day,
        timezone_name=timezone_name,
        session_start=session_start,
        session_end=session_end,
        records=records,
        progress=progress,
    )

    result = DayDivergenceResult(
        day=day,
        sample_timestamps=len(records),
        exact_timestamp_hits=0,
        nearby_hits_within_tolerance=0,
        missing_without_nearby_match=0,
        exact_top10_matches=0,
        exact_bbo_matches=0,
        nearby_top10_matches=0,
        nearby_bbo_matches=0,
        category_counts={},
        price_level_mismatch_counts={},
        size_level_mismatch_counts={},
        example_diffs=[],
    )
    tolerance_ns = tolerance_us * 1_000

    for record in records:
        if record.exact is not None:
            result.exact_timestamp_hits += 1
            candidate = record.exact
            match_mode = "exact"
            delta_us = 0.0
        else:
            match_mode, candidate, delta_us = choose_nearest_candidate(record, tolerance_ns)
            if candidate is None or match_mode is None:
                result.missing_without_nearby_match += 1
                result.category_counts["missing_without_nearby_match"] = result.category_counts.get("missing_without_nearby_match", 0) + 1
                if len(result.example_diffs) < examples:
                    result.example_diffs.append(
                        ExampleDiff(
                            timestamp=record.local_ts,
                            match_mode="missing",
                            matched_timestamp=None,
                            delta_us=delta_us,
                            category="missing_without_nearby_match",
                            processed_bbo=bbo_to_dict(record.processed_snapshot),
                            matched_bbo=None,
                            changed_levels=[],
                        )
                    )
                continue
            result.nearby_hits_within_tolerance += 1

        if snapshot_bbo(record.processed_snapshot) == snapshot_bbo(candidate.snapshot):
            if match_mode == "exact":
                result.exact_bbo_matches += 1
            else:
                result.nearby_bbo_matches += 1

        category, changed_levels, price_counts, size_counts = analyze_snapshot_difference(
            record.processed_snapshot,
            candidate.snapshot,
        )
        result.category_counts[category] = result.category_counts.get(category, 0) + 1
        for key, value in price_counts.items():
            result.price_level_mismatch_counts[key] = result.price_level_mismatch_counts.get(key, 0) + value
        for key, value in size_counts.items():
            result.size_level_mismatch_counts[key] = result.size_level_mismatch_counts.get(key, 0) + value

        if category == "exact_top10_match":
            if match_mode == "exact":
                result.exact_top10_matches += 1
            else:
                result.nearby_top10_matches += 1

        if category != "exact_top10_match" and len(result.example_diffs) < examples:
            matched_local_ts = None
            if candidate is not None:
                matched_local_ts = candidate.raw_ts
            result.example_diffs.append(
                ExampleDiff(
                    timestamp=record.local_ts,
                    match_mode=match_mode,
                    matched_timestamp=matched_local_ts,
                    delta_us=delta_us,
                    category=category,
                    processed_bbo=bbo_to_dict(record.processed_snapshot),
                    matched_bbo=bbo_to_dict(candidate.snapshot) if candidate is not None else None,
                    changed_levels=changed_levels,
                )
            )

    return result


def write_analysis_report(symbol: str, analysis_root: Path, results: List[DayDivergenceResult]) -> Path:
    output_dir = analysis_root / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mbp10_divergence_analysis.json"
    with output_path.open("w") as handle:
        json.dump(
            {
                "symbol": symbol,
                "days": [asdict(result) for result in results],
            },
            handle,
            indent=2,
        )
    return output_path


def analyze_symbol(config: DivergenceConfig, progress: Optional[DatabentoProgress] = None) -> List[DayDivergenceResult]:
    jobs = discover_schema_jobs_for_symbol(config.raw_root, config.symbol)
    if "mbp-10" not in jobs:
        raise RuntimeError("Missing required raw schema directory: mbp-10")

    mbp_files = collect_schema_files(jobs["mbp-10"].directory, ".mbp-10")
    selected_days: List[tuple[str, Path]] = []
    for key, path in sorted(mbp_files.items()):
        day = key.split("-")[-1]
        if config.days is not None and day not in config.days:
            continue
        selected_days.append((day, path))

    overall_task = None
    if progress is not None:
        overall_task = progress.add_task(f"[cyan]Analyzing {len(selected_days)} day(s)[/cyan]", total=len(selected_days))

    results: List[DayDivergenceResult] = []
    for day, mbp_path in selected_days:
        result = analyze_day(
            symbol=config.symbol,
            day=day,
            processed_root=config.processed_root,
            mbp_path=mbp_path,
            timezone_name=config.timezone_name,
            session_start=config.session_start,
            session_end=config.session_end,
            sample_every=config.sample_every,
            edge_samples=config.edge_samples,
            tolerance_us=config.tolerance_us,
            examples=config.examples,
            progress=progress,
        )
        results.append(result)
        if progress is not None:
            progress.update(overall_task, advance=1)
            progress.print_summary(
                f"[bold green]Analyzed {day}[/bold green]: "
                f"exact_hits={result.exact_timestamp_hits:,} "
                f"nearby_hits={result.nearby_hits_within_tolerance:,} "
                f"missing={result.missing_without_nearby_match:,} "
                f"exact_bbo={result.exact_bbo_matches:,} "
                f"nearby_bbo={result.nearby_bbo_matches:,}"
            )

    report_path = write_analysis_report(config.symbol, config.analysis_root, results)
    if progress is not None:
        progress.print_summary(f"[bold blue]Wrote divergence report[/bold blue] {report_path}")
    return results
