from __future__ import annotations

import csv
import json
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from .common import (
    csv_reader,
    discover_schema_jobs_for_symbol,
    localize_timestamp,
    parse_session_time,
    price_to_int,
)
from .convert import collect_schema_files
from .progress import DatabentoProgress


LEVELS = 10
ROW_UPDATE_INTERVAL = 100_000
Snapshot = Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]


@dataclass
class ValidationConfig:
    raw_root: Path
    processed_root: Path
    validation_root: Path
    symbol: str
    timezone_name: str = "America/New_York"
    session_start: str = "09:30:00"
    session_end: str = "16:00:00"
    sample_every: int = 10_000
    edge_samples: int = 1_000
    days: Optional[Set[str]] = None

    @property
    def session_start_time(self) -> time:
        return parse_session_time(self.session_start)

    @property
    def session_end_time(self) -> time:
        return parse_session_time(self.session_end)


@dataclass
class DayValidationResult:
    day: str
    processed_emitted_timestamps: int
    processed_book_change_timestamps: int
    mbp_timestamps: int
    sample_timestamps: int
    matched_sample_timestamps: int
    missing_sample_timestamps: int
    exact_bbo_matches: int
    exact_top10_matches: int
    processed_first_timestamp: Optional[str]
    processed_last_timestamp: Optional[str]
    processed_book_change_first_timestamp: Optional[str]
    processed_book_change_last_timestamp: Optional[str]
    mbp_first_timestamp: Optional[str]
    mbp_last_timestamp: Optional[str]
    first_mismatch_timestamp: Optional[str]
    first_mismatch_reason: Optional[str]

    @property
    def count_match(self) -> bool:
        return self.processed_book_change_timestamps == self.mbp_timestamps

    @property
    def first_timestamp_match(self) -> bool:
        return self.processed_book_change_first_timestamp == self.mbp_first_timestamp

    @property
    def last_timestamp_match(self) -> bool:
        return self.processed_book_change_last_timestamp == self.mbp_last_timestamp

    @property
    def bbo_match_rate(self) -> float:
        if self.matched_sample_timestamps == 0:
            return 0.0
        return self.exact_bbo_matches / self.matched_sample_timestamps

    @property
    def top10_match_rate(self) -> float:
        if self.matched_sample_timestamps == 0:
            return 0.0
        return self.exact_top10_matches / self.matched_sample_timestamps

    @property
    def sample_coverage_rate(self) -> float:
        if self.sample_timestamps == 0:
            return 0.0
        return self.matched_sample_timestamps / self.sample_timestamps


@dataclass
class ProcessedSampleSet:
    emitted_count: int
    book_change_count: int
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]
    book_change_first_timestamp: Optional[str]
    book_change_last_timestamp: Optional[str]
    samples: "OrderedDict[str, Snapshot]"


@dataclass
class MbpScanResult:
    count: int
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]
    sampled_snapshots: Dict[str, Snapshot]


def normalize_book_side(row: dict[str, str], prefix: str) -> Tuple[Tuple[int, int], ...]:
    levels: List[Tuple[int, int]] = []
    for level in range(LEVELS):
        price_key = f"{prefix}_px_{level:02d}"
        size_key = f"{prefix}_sz_{level:02d}"
        price_value = row[price_key]
        size_value = row[size_key]
        if not price_value:
            return ()
        levels.append((price_to_int(price_value), int(size_value)))
    return tuple(levels)


def snapshot_bbo(snapshot: Snapshot) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    asks, bids = snapshot
    return asks[0], bids[0]


def first_mismatch_reason(processed: Snapshot, mbp: Snapshot) -> str:
    if snapshot_bbo(processed) != snapshot_bbo(mbp):
        return "bbo"
    return "depth"


def local_timestamp_to_utc_event(timestamp: str, timezone_name: str) -> str:
    if "." in timestamp:
        base_part, fractional = timestamp.split(".", 1)
    else:
        base_part, fractional = timestamp, ""
    local_dt = datetime.fromisoformat(base_part).replace(tzinfo=ZoneInfo(timezone_name))
    utc_dt = local_dt.astimezone(timezone.utc)
    if fractional:
        return f"{utc_dt:%Y-%m-%dT%H:%M:%S}.{fractional}Z"
    return f"{utc_dt:%Y-%m-%dT%H:%M:%SZ}"


def utc_session_bounds(day: str, timezone_name: str, session_start: time, session_end: time) -> tuple[str, str]:
    local_tz = ZoneInfo(timezone_name)
    trading_day = datetime.strptime(day, "%Y%m%d").date()
    start_utc = datetime.combine(trading_day, session_start, tzinfo=local_tz).astimezone(timezone.utc)
    end_utc = datetime.combine(trading_day, session_end, tzinfo=local_tz).astimezone(timezone.utc)
    return start_utc.strftime("%Y-%m-%dT%H:%M:%S"), end_utc.strftime("%Y-%m-%dT%H:%M:%S")


def build_processed_sample_set(day_dir: Path, sample_every: int, edge_samples: int) -> ProcessedSampleSet:
    front: "OrderedDict[int, Tuple[str, Snapshot]]" = OrderedDict()
    middle: "OrderedDict[int, Tuple[str, Snapshot]]" = OrderedDict()
    tail: deque[Tuple[int, str, Snapshot]] = deque(maxlen=edge_samples)
    emitted_count = 0
    book_change_count = 0
    first_timestamp = None
    last_timestamp = None
    book_change_first_timestamp = None
    book_change_last_timestamp = None
    samples: "OrderedDict[str, Snapshot]" = OrderedDict()
    ask_path = day_dir / "ask.csv"
    bid_path = day_dir / "bid.csv"
    previous_snapshot: Optional[Snapshot] = None
    with ask_path.open(newline="") as ask_handle, bid_path.open(newline="") as bid_handle:
        ask_reader = csv.DictReader(ask_handle)
        bid_reader = csv.DictReader(bid_handle)
        for emitted_count, (ask_row, bid_row) in enumerate(zip(ask_reader, bid_reader), start=1):
            timestamp = ask_row["timestamp"]
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp
            asks = tuple(
                (price_to_int(ask_row[f"ask{level}_price"]), int(ask_row[f"ask{level}_volume"]))
                for level in range(1, LEVELS + 1)
            )
            bids = tuple(
                (price_to_int(bid_row[f"bid{level}_price"]), int(bid_row[f"bid{level}_volume"]))
                for level in range(1, LEVELS + 1)
            )
            snapshot = (asks, bids)
            if snapshot == previous_snapshot:
                continue
            previous_snapshot = snapshot
            book_change_count += 1
            if book_change_first_timestamp is None:
                book_change_first_timestamp = timestamp
            book_change_last_timestamp = timestamp
            if book_change_count <= edge_samples:
                front[book_change_count] = (timestamp, snapshot)
            if sample_every > 0 and book_change_count % sample_every == 0:
                middle[book_change_count] = (timestamp, snapshot)
            tail.append((book_change_count, timestamp, snapshot))

    sample_rows: "OrderedDict[int, Tuple[str, Snapshot]]" = OrderedDict()
    for group in (front.items(), middle.items(), tail):
        for item in group:
            if len(item) == 2:
                row_index, payload = item
            else:
                row_index, timestamp, snapshot = item
                payload = (timestamp, snapshot)
            sample_rows[row_index] = payload

    for _, (timestamp, snapshot) in sample_rows.items():
        samples[timestamp] = snapshot

    return ProcessedSampleSet(
        emitted_count=emitted_count,
        book_change_count=book_change_count,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        book_change_first_timestamp=book_change_first_timestamp,
        book_change_last_timestamp=book_change_last_timestamp,
        samples=samples,
    )


def scan_mbp_samples(
    path: Path,
    day: str,
    timezone_name: str,
    session_start: time,
    session_end: time,
    sample_utc_map: Dict[str, str],
    on_row: Optional[Callable[[int], None]] = None,
) -> MbpScanResult:
    current_ts: Optional[str] = None
    current_snapshot: Optional[Snapshot] = None
    first_raw_ts: Optional[str] = None
    last_raw_ts: Optional[str] = None
    unique_count = 0
    sampled_snapshots: Dict[str, Snapshot] = {}
    start_prefix, end_prefix = utc_session_bounds(day, timezone_name, session_start, session_end)
    last_idx = 0

    def finalize_group(raw_ts: Optional[str], snapshot: Optional[Snapshot]) -> None:
        nonlocal unique_count, first_raw_ts, last_raw_ts
        if raw_ts is None or snapshot is None:
            return
        unique_count += 1
        if first_raw_ts is None:
            first_raw_ts = raw_ts
        last_raw_ts = raw_ts
        local_ts = sample_utc_map.get(raw_ts)
        if local_ts is not None:
            sampled_snapshots[local_ts] = snapshot

    for idx, row in enumerate(csv_reader(path), start=1):
        last_idx = idx
        raw_ts = row["ts_event"]
        raw_prefix = raw_ts[:19]
        if raw_prefix > end_prefix:
            break
        if raw_prefix < start_prefix:
            if on_row is not None and idx % ROW_UPDATE_INTERVAL == 0:
                on_row(idx)
            continue

        bids = normalize_book_side(row, "bid")
        asks = normalize_book_side(row, "ask")
        if not bids or not asks:
            if on_row is not None and idx % ROW_UPDATE_INTERVAL == 0:
                on_row(idx)
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

        if on_row is not None and idx % ROW_UPDATE_INTERVAL == 0:
            on_row(idx)

    finalize_group(current_ts, current_snapshot)
    if on_row is not None:
        on_row(last_idx)

    first_timestamp = localize_timestamp(first_raw_ts, timezone_name)[1] if first_raw_ts else None
    last_timestamp = localize_timestamp(last_raw_ts, timezone_name)[1] if last_raw_ts else None
    return MbpScanResult(
        count=unique_count,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        sampled_snapshots=sampled_snapshots,
    )


def compare_day(
    processed_day_dir: Path,
    mbp_path: Path,
    timezone_name: str,
    session_start: time,
    session_end: time,
    sample_every: int,
    edge_samples: int,
    progress: Optional[DatabentoProgress] = None,
    day: Optional[str] = None,
) -> DayValidationResult:
    processed_samples = build_processed_sample_set(
        processed_day_dir,
        sample_every=sample_every,
        edge_samples=edge_samples,
    )
    sample_utc_map = {
        local_timestamp_to_utc_event(timestamp, timezone_name): timestamp
        for timestamp in processed_samples.samples
    }

    mbp_task = None
    if progress is not None and day is not None:
        mbp_task = progress.add_task(f"[magenta]{day}[/magenta] MBP-10 scan", total=None)

    mbp_scan = scan_mbp_samples(
        path=mbp_path,
        day=day or processed_day_dir.name,
        timezone_name=timezone_name,
        session_start=session_start,
        session_end=session_end,
        sample_utc_map=sample_utc_map,
        on_row=(
            (lambda count, task_id=mbp_task: progress.update(task_id, description=f"[magenta]{day}[/magenta] MBP-10 rows: {count:,}"))
            if progress is not None and day is not None
            else None
        ),
    )

    exact_bbo_matches = 0
    exact_top10_matches = 0
    first_mismatch_timestamp = None
    first_mismatch_reason_value = None

    for timestamp, processed_snapshot in processed_samples.samples.items():
        mbp_snapshot = mbp_scan.sampled_snapshots.get(timestamp)
        if mbp_snapshot is None:
            if first_mismatch_timestamp is None:
                first_mismatch_timestamp = timestamp
                first_mismatch_reason_value = "missing_sample_timestamp"
            continue
        if snapshot_bbo(processed_snapshot) == snapshot_bbo(mbp_snapshot):
            exact_bbo_matches += 1
        if processed_snapshot == mbp_snapshot:
            exact_top10_matches += 1
        elif first_mismatch_timestamp is None:
            first_mismatch_timestamp = timestamp
            first_mismatch_reason_value = first_mismatch_reason(processed_snapshot, mbp_snapshot)

    if first_mismatch_timestamp is None and processed_samples.book_change_count != mbp_scan.count:
        first_mismatch_timestamp = processed_samples.book_change_first_timestamp or mbp_scan.first_timestamp
        first_mismatch_reason_value = "timestamp_count_mismatch"
    if first_mismatch_timestamp is None and processed_samples.book_change_first_timestamp != mbp_scan.first_timestamp:
        first_mismatch_timestamp = processed_samples.book_change_first_timestamp or mbp_scan.first_timestamp
        first_mismatch_reason_value = "first_timestamp_mismatch"
    if first_mismatch_timestamp is None and processed_samples.book_change_last_timestamp != mbp_scan.last_timestamp:
        first_mismatch_timestamp = processed_samples.book_change_last_timestamp or mbp_scan.last_timestamp
        first_mismatch_reason_value = "last_timestamp_mismatch"

    if progress is not None and mbp_task is not None:
        progress.update(mbp_task, description=f"[magenta]{day}[/magenta] MBP-10 validated", completed=1, total=1)
        progress.remove_task(mbp_task)

    return DayValidationResult(
        day=day or processed_day_dir.name,
        processed_emitted_timestamps=processed_samples.emitted_count,
        processed_book_change_timestamps=processed_samples.book_change_count,
        mbp_timestamps=mbp_scan.count,
        sample_timestamps=len(processed_samples.samples),
        matched_sample_timestamps=len(mbp_scan.sampled_snapshots),
        missing_sample_timestamps=len(processed_samples.samples) - len(mbp_scan.sampled_snapshots),
        exact_bbo_matches=exact_bbo_matches,
        exact_top10_matches=exact_top10_matches,
        processed_first_timestamp=processed_samples.first_timestamp,
        processed_last_timestamp=processed_samples.last_timestamp,
        processed_book_change_first_timestamp=processed_samples.book_change_first_timestamp,
        processed_book_change_last_timestamp=processed_samples.book_change_last_timestamp,
        mbp_first_timestamp=mbp_scan.first_timestamp,
        mbp_last_timestamp=mbp_scan.last_timestamp,
        first_mismatch_timestamp=first_mismatch_timestamp,
        first_mismatch_reason=first_mismatch_reason_value,
    )


def write_validation_report(symbol: str, validation_root: Path, results: List[DayValidationResult]) -> Path:
    output_dir = validation_root / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mbp10_validation.json"
    payload = {
        "symbol": symbol,
        "days": [
            asdict(result)
            | {
                "count_match": result.count_match,
                "first_timestamp_match": result.first_timestamp_match,
                "last_timestamp_match": result.last_timestamp_match,
                "sample_coverage_rate": result.sample_coverage_rate,
                "bbo_match_rate": result.bbo_match_rate,
                "top10_match_rate": result.top10_match_rate,
            }
            for result in results
        ],
    }
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def validate_symbol(config: ValidationConfig, progress: Optional[DatabentoProgress] = None) -> List[DayValidationResult]:
    jobs = discover_schema_jobs_for_symbol(config.raw_root, config.symbol)
    missing = {"mbp-10"} - set(jobs)
    if missing:
        raise RuntimeError(f"Missing required raw schema directories: {sorted(missing)}")

    mbp_files = collect_schema_files(jobs["mbp-10"].directory, ".mbp-10")
    selected_days: List[Tuple[str, Path]] = []
    for key, path in sorted(mbp_files.items()):
        day = key.split("-")[-1]
        if config.days is not None and day not in config.days:
            continue
        selected_days.append((day, path))

    overall_task = None
    if progress is not None:
        overall_task = progress.add_task(f"[cyan]Validating {len(selected_days)} day(s)[/cyan]", total=len(selected_days))

    results: List[DayValidationResult] = []
    for day, mbp_path in selected_days:
        processed_day_dir = config.processed_root / config.symbol / day
        if not processed_day_dir.exists():
            raise RuntimeError(f"Missing processed output for {config.symbol} {day}: {processed_day_dir}")
        result = compare_day(
            processed_day_dir=processed_day_dir,
            mbp_path=mbp_path,
            timezone_name=config.timezone_name,
            session_start=config.session_start_time,
            session_end=config.session_end_time,
            sample_every=config.sample_every,
            edge_samples=config.edge_samples,
            progress=progress,
            day=day,
        )
        results.append(result)
        if progress is not None:
            progress.update(overall_task, advance=1)
            progress.print_summary(
                f"[bold green]Validated {day}[/bold green]: "
                f"count_match={result.count_match} "
                f"boundary_match={result.first_timestamp_match and result.last_timestamp_match} "
                f"samples={result.sample_timestamps:,} "
                f"sample_hits={result.matched_sample_timestamps:,} "
                f"sample_coverage={result.sample_coverage_rate:.6f} "
                f"bbo_match_rate={result.bbo_match_rate:.6f} "
                f"top10_match_rate={result.top10_match_rate:.6f}"
            )

    report_path = write_validation_report(config.symbol, config.validation_root, results)
    if progress is not None:
        progress.print_summary(f"[bold blue]Wrote validation report[/bold blue] {report_path}")
    return results
