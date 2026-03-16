from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from .book import OrderBook
from .common import (
    csv_reader,
    discover_schema_jobs_for_symbol,
    int_to_price,
    localize_timestamp,
    parse_session_time,
    price_to_int,
    should_keep_timestamp,
)
from .progress import DatabentoProgress


MSG_COLUMNS = [
    "market_buy_volume",
    "market_buy_n",
    "market_sell_volume",
    "market_sell_n",
    "limit_buy_volume",
    "limit_buy_n",
    "limit_sell_volume",
    "limit_sell_n",
    "withdraw_buy_volume",
    "withdraw_buy_n",
    "withdraw_sell_volume",
    "withdraw_sell_n",
]

ROW_UPDATE_INTERVAL = 100_000


@dataclass
class ConversionConfig:
    raw_root: Path
    output_root: Path
    symbol: str
    timezone_name: str = "America/New_York"
    session_start: str = "09:30:00"
    session_end: str = "16:00:00"
    days: Optional[Set[str]] = None

    @property
    def session_start_time(self) -> time:
        return parse_session_time(self.session_start)

    @property
    def session_end_time(self) -> time:
        return parse_session_time(self.session_end)


def zero_msg_counts() -> Dict[str, int]:
    return {column: 0 for column in MSG_COLUMNS}


def collect_schema_files(directory: Path, stem_suffix: str) -> Dict[str, Path]:
    files_by_stem: Dict[str, Path] = {}
    for path in sorted(directory.glob(f"*{stem_suffix}.csv.zst")):
        stem = path.name.replace(".csv.zst", "")
        key = stem.removesuffix(stem_suffix)
        files_by_stem[key] = path
    for path in sorted(directory.glob(f"*{stem_suffix}.csv")):
        stem = path.name.replace(".csv", "")
        key = stem.removesuffix(stem_suffix)
        files_by_stem[key] = path
    return files_by_stem


def infer_trade_side(row: dict[str, str], tbbo_row: Optional[dict[str, str]]) -> str:
    side = row["side"]
    if side in {"A", "B"}:
        return side
    if tbbo_row is None:
        return "N"
    price = price_to_int(row["price"])
    bid = price_to_int(tbbo_row["bid_px_00"])
    ask = price_to_int(tbbo_row["ask_px_00"])
    if price == ask:
        return "B"
    if price == bid:
        return "A"
    return "N"


def load_tbbo_by_sequence(
    tbbo_path: Path,
    on_row: Optional[Callable[[int], None]] = None,
) -> Dict[str, dict[str, str]]:
    tbbo_by_sequence: Dict[str, dict[str, str]] = {}
    for idx, row in enumerate(csv_reader(tbbo_path), start=1):
        tbbo_by_sequence[row["sequence"]] = row
        if on_row is not None and idx % ROW_UPDATE_INTERVAL == 0:
            on_row(idx)
    if on_row is not None:
        on_row(len(tbbo_by_sequence))
    return tbbo_by_sequence


def build_trade_maps(
    trades_path: Path,
    tbbo_path: Path,
    timezone_name: str,
    session_start: time,
    session_end: time,
    on_tbbo_row: Optional[Callable[[int], None]] = None,
    on_trade_row: Optional[Callable[[int], None]] = None,
) -> tuple[List[dict[str, str]], Dict[str, Dict[str, int]]]:
    tbbo_by_sequence = load_tbbo_by_sequence(tbbo_path, on_row=on_tbbo_row)
    trade_rows: List[dict[str, str]] = []
    market_by_ts: Dict[str, Dict[str, int]] = defaultdict(zero_msg_counts)
    last_price: Optional[int] = None
    last_inferred_side = "N"
    last_non_n_side: Optional[str] = None

    for idx, row in enumerate(csv_reader(trades_path), start=1):
        local_dt, local_ts = localize_timestamp(row["ts_event"], timezone_name)
        if not should_keep_timestamp(local_dt, session_start, session_end):
            if on_trade_row is not None and idx % ROW_UPDATE_INTERVAL == 0:
                on_trade_row(idx)
            continue

        trade_price = price_to_int(row["price"])
        tbbo_row = tbbo_by_sequence.get(row["sequence"])
        side = infer_trade_side(row, tbbo_row)
        if side == "N" and last_price is not None:
            if trade_price > last_price:
                side = "B"
            elif trade_price < last_price:
                side = "A"
            elif last_non_n_side in {"A", "B"}:
                side = last_non_n_side
            elif last_inferred_side in {"A", "B"}:
                side = last_inferred_side
        if side == "N" and tbbo_row is not None:
            bid = price_to_int(tbbo_row["bid_px_00"])
            ask = price_to_int(tbbo_row["ask_px_00"])
            dist_to_bid = abs(trade_price - bid)
            dist_to_ask = abs(ask - trade_price)
            if dist_to_ask < dist_to_bid:
                side = "B"
            elif dist_to_bid < dist_to_ask:
                side = "A"
            elif last_non_n_side in {"A", "B"}:
                side = last_non_n_side
            else:
                midpoint = bid + (ask - bid) / 2
                side = "B" if trade_price >= midpoint else "A"

        trade_rows.append(
            {
                "timestamp": local_ts,
                "price": row["price"],
                "size": row["size"],
                "aggressor_side": side,
            }
        )
        if side == "B":
            market_by_ts[local_ts]["market_buy_volume"] += int(row["size"])
            market_by_ts[local_ts]["market_buy_n"] += 1
        elif side == "A":
            market_by_ts[local_ts]["market_sell_volume"] += int(row["size"])
            market_by_ts[local_ts]["market_sell_n"] += 1
        last_price = trade_price
        last_inferred_side = side
        if side in {"A", "B"}:
            last_non_n_side = side
        if on_trade_row is not None and idx % ROW_UPDATE_INTERVAL == 0:
            on_trade_row(idx)

    if on_trade_row is not None:
        on_trade_row(idx if 'idx' in locals() else 0)

    return trade_rows, market_by_ts


def write_trades(trade_rows: List[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trades.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "price", "size", "aggressor_side"])
        writer.writeheader()
        writer.writerows(trade_rows)


def ask_header(levels: int = 10) -> List[str]:
    header = ["timestamp"]
    for level in range(1, levels + 1):
        header.extend([f"ask{level}_price", f"ask{level}_volume"])
    return header


def bid_header(levels: int = 10) -> List[str]:
    header = ["timestamp"]
    for level in range(1, levels + 1):
        header.extend([f"bid{level}_price", f"bid{level}_volume"])
    return header


def process_mbo_day(
    mbo_path: Path,
    output_dir: Path,
    timezone_name: str,
    session_start: time,
    session_end: time,
    market_by_ts: Dict[str, Dict[str, int]],
    on_mbo_row: Optional[Callable[[int], None]] = None,
    on_emit: Optional[Callable[[int], None]] = None,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ask_path = output_dir / "ask.csv"
    bid_path = output_dir / "bid.csv"
    price_path = output_dir / "price.csv"
    msg_path = output_dir / "msg.csv"

    book = OrderBook()
    previous_signature = None
    emitted_timestamps = set()
    current_sequence: Optional[str] = None
    pending_fill_cancels: set[tuple[str, int, str, int]] = set()

    stats = {
        "rows_emitted": 0,
        "trade_timestamps_with_rows": 0,
        "unmatched_trade_timestamps": 0,
    }

    with ask_path.open("w", newline="") as ask_handle, bid_path.open("w", newline="") as bid_handle, price_path.open(
        "w", newline=""
    ) as price_handle, msg_path.open("w", newline="") as msg_handle:
        ask_writer = csv.writer(ask_handle)
        bid_writer = csv.writer(bid_handle)
        price_writer = csv.writer(price_handle)
        msg_writer = csv.writer(msg_handle)

        ask_writer.writerow(ask_header())
        bid_writer.writerow(bid_header())
        price_writer.writerow(["timestamp", "ask1_price", "bid1_price", "midprice"])
        msg_writer.writerow(["timestamp", *MSG_COLUMNS])

        current_ts: Optional[str] = None
        current_counts = zero_msg_counts()

        def flush_group(group_ts: Optional[str], counts: Dict[str, int]) -> None:
            nonlocal previous_signature
            if group_ts is None:
                return
            snapshot = book.top_n(10)
            if snapshot is None:
                return
            market_counts = market_by_ts.get(group_ts, zero_msg_counts())
            merged_counts = {column: counts[column] + market_counts[column] for column in MSG_COLUMNS}
            signature = snapshot
            should_emit = signature != previous_signature or any(merged_counts.values())
            if not should_emit:
                return

            asks, bids = snapshot
            ask_row = [group_ts]
            bid_row = [group_ts]
            for price, size in asks:
                ask_row.extend([int_to_price(price), size])
            for price, size in bids:
                bid_row.extend([int_to_price(price), size])

            best_ask = asks[0][0]
            best_bid = bids[0][0]
            mid = (best_ask + best_bid) / 2

            ask_writer.writerow(ask_row)
            bid_writer.writerow(bid_row)
            price_writer.writerow([group_ts, int_to_price(best_ask), int_to_price(best_bid), f"{mid / 1_000_000_000:.9f}".rstrip("0").rstrip(".")])
            msg_writer.writerow([group_ts, *[merged_counts[column] for column in MSG_COLUMNS]])

            emitted_timestamps.add(group_ts)
            stats["rows_emitted"] += 1
            previous_signature = signature
            if on_emit is not None and stats["rows_emitted"] % 5_000 == 0:
                on_emit(stats["rows_emitted"])

        for idx, row in enumerate(csv_reader(mbo_path), start=1):
            local_dt, local_ts = localize_timestamp(row["ts_event"], timezone_name)
            action = row["action"]
            side = row["side"]
            size = int(row["size"])
            order_id = int(row["order_id"])
            sequence = row["sequence"]
            in_session = should_keep_timestamp(local_dt, session_start, session_end)

            if sequence != current_sequence:
                current_sequence = sequence
                pending_fill_cancels.clear()

            if local_dt.time() > session_end:
                flush_group(current_ts, current_counts)
                break

            if in_session:
                if current_ts is None:
                    current_ts = local_ts
                elif local_ts != current_ts:
                    flush_group(current_ts, current_counts)
                    current_ts = local_ts
                    current_counts = zero_msg_counts()

            if action == "R":
                book.clear()
            elif action == "A" and side in {"A", "B"}:
                price = price_to_int(row["price"])
                book.add(order_id, side, price, size)
                if in_session and side == "B":
                    current_counts["limit_buy_volume"] += size
                    current_counts["limit_buy_n"] += 1
                elif in_session and side == "A":
                    current_counts["limit_sell_volume"] += size
                    current_counts["limit_sell_n"] += 1
            elif action == "C":
                fill_cancel_key = (sequence, order_id, side, size)
                if fill_cancel_key in pending_fill_cancels:
                    pending_fill_cancels.remove(fill_cancel_key)
                    if on_mbo_row is not None and idx % ROW_UPDATE_INTERVAL == 0:
                        on_mbo_row(idx)
                    continue
                cancelled = book.cancel(order_id, size)
                if in_session and cancelled is not None:
                    cancelled_side, removed = cancelled
                    if cancelled_side == "B":
                        current_counts["withdraw_buy_volume"] += removed
                        current_counts["withdraw_buy_n"] += 1
                    else:
                        current_counts["withdraw_sell_volume"] += removed
                        current_counts["withdraw_sell_n"] += 1
            elif action == "F":
                book.fill(order_id, size)
                if side in {"A", "B"} and size > 0:
                    pending_fill_cancels.add((sequence, order_id, side, size))
            elif action == "M":
                price = price_to_int(row["price"])
                modified = book.modify(order_id, price, size)
                if in_session and modified is not None:
                    modified_side, old_size, new_size = modified
                    if old_size > 0:
                        if modified_side == "B":
                            current_counts["withdraw_buy_volume"] += old_size
                            current_counts["withdraw_buy_n"] += 1
                        else:
                            current_counts["withdraw_sell_volume"] += old_size
                            current_counts["withdraw_sell_n"] += 1
                    if new_size > 0:
                        if modified_side == "B":
                            current_counts["limit_buy_volume"] += new_size
                            current_counts["limit_buy_n"] += 1
                        else:
                            current_counts["limit_sell_volume"] += new_size
                            current_counts["limit_sell_n"] += 1
            if on_mbo_row is not None and idx % ROW_UPDATE_INTERVAL == 0:
                on_mbo_row(idx)

        flush_group(current_ts, current_counts)
        if on_mbo_row is not None:
            on_mbo_row(idx if 'idx' in locals() else 0)
        if on_emit is not None:
            on_emit(stats["rows_emitted"])

    stats["trade_timestamps_with_rows"] = sum(1 for ts in market_by_ts if ts in emitted_timestamps)
    stats["unmatched_trade_timestamps"] = sum(1 for ts in market_by_ts if ts not in emitted_timestamps)
    return stats


def convert_symbol(config: ConversionConfig, progress: Optional[DatabentoProgress] = None) -> List[dict[str, int]]:
    jobs = discover_schema_jobs_for_symbol(config.raw_root, config.symbol)
    missing = {"mbo", "trades", "tbbo"} - set(jobs)
    if missing:
        raise RuntimeError(f"Missing required raw schema directories: {sorted(missing)}")

    mbo_files = collect_schema_files(jobs["mbo"].directory, ".mbo")
    trades_files = collect_schema_files(jobs["trades"].directory, ".trades")
    tbbo_files = collect_schema_files(jobs["tbbo"].directory, ".tbbo")

    selected_files = []
    for key, mbo_path in sorted(mbo_files.items()):
        day = key.split("-")[-1]
        if config.days is not None and day not in config.days:
            continue
        selected_files.append((mbo_path, key, day))

    overall_task = None
    if progress is not None:
        overall_task = progress.add_task(f"[cyan]Converting {len(selected_files)} day(s)[/cyan]", total=len(selected_files))

    results: List[dict[str, int]] = []
    for mbo_path, key, day in selected_files:
        trades_path = trades_files.get(key)
        tbbo_path = tbbo_files.get(key)
        if trades_path is None or tbbo_path is None:
            raise RuntimeError(f"Could not find matching trades/tbbo files for {mbo_path.name}")

        tbbo_task = trade_task = mbo_task = emit_task = None
        if progress is not None:
            tbbo_task = progress.add_task(f"[magenta]{day}[/magenta] TBBO rows", total=None)
            trade_task = progress.add_task(f"[blue]{day}[/blue] Trades rows", total=None)
            mbo_task = progress.add_task(f"[yellow]{day}[/yellow] MBO rows", total=None)
            emit_task = progress.add_task(f"[green]{day}[/green] Output rows", total=None)

        trade_rows, market_by_ts = build_trade_maps(
            trades_path=trades_path,
            tbbo_path=tbbo_path,
            timezone_name=config.timezone_name,
            session_start=config.session_start_time,
            session_end=config.session_end_time,
            on_tbbo_row=(lambda count, task_id=tbbo_task: progress.update(task_id, description=f"[magenta]{day}[/magenta] TBBO rows: {count:,}")) if progress is not None else None,
            on_trade_row=(lambda count, task_id=trade_task: progress.update(task_id, description=f"[blue]{day}[/blue] Trades rows: {count:,}")) if progress is not None else None,
        )
        if progress is not None:
            progress.update(tbbo_task, description=f"[magenta]{day}[/magenta] TBBO loaded", completed=1, total=1)
            progress.update(trade_task, description=f"[blue]{day}[/blue] Trades prepared: {len(trade_rows):,}", completed=1, total=1)

        output_dir = config.output_root / config.symbol / day
        write_trades(trade_rows, output_dir)
        stats = process_mbo_day(
            mbo_path=mbo_path,
            output_dir=output_dir,
            timezone_name=config.timezone_name,
            session_start=config.session_start_time,
            session_end=config.session_end_time,
            market_by_ts=market_by_ts,
            on_mbo_row=(lambda count, task_id=mbo_task: progress.update(task_id, description=f"[yellow]{day}[/yellow] MBO rows: {count:,}")) if progress is not None else None,
            on_emit=(lambda count, task_id=emit_task: progress.update(task_id, description=f"[green]{day}[/green] Output rows: {count:,}")) if progress is not None else None,
        )
        stats["trade_rows"] = len(trade_rows)
        stats["day"] = int(day)
        results.append(stats)
        if progress is not None:
            progress.update(mbo_task, description=f"[yellow]{day}[/yellow] MBO processed", completed=1, total=1)
            progress.update(emit_task, description=f"[green]{day}[/green] Output rows: {stats['rows_emitted']:,}", completed=1, total=1)
            progress.update(overall_task, advance=1)
            progress.print_summary(
                f"[bold green]Converted {day}[/bold green]: "
                f"trades={stats['trade_rows']:,} rows_emitted={stats['rows_emitted']:,} "
                f"matched_trade_timestamps={stats['trade_timestamps_with_rows']:,} "
                f"unmatched_trade_timestamps={stats['unmatched_trade_timestamps']:,}"
            )
            progress.remove_task(tbbo_task)
            progress.remove_task(trade_task)
            progress.remove_task(mbo_task)
            progress.remove_task(emit_task)
    return results
