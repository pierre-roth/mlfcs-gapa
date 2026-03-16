from __future__ import annotations

import csv
import json
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, TextIO
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class SchemaJob:
    schema: str
    directory: Path


def load_metadata(metadata_path: Path) -> dict:
    with metadata_path.open() as handle:
        return json.load(handle)


def discover_all_schema_jobs(raw_root: Path) -> List[SchemaJob]:
    jobs: List[SchemaJob] = []
    for metadata_path in sorted(raw_root.glob("*/metadata.json")):
        metadata = load_metadata(metadata_path)
        schema = metadata["query"]["schema"]
        jobs.append(SchemaJob(schema=schema, directory=metadata_path.parent))
    return jobs


def discover_schema_jobs(raw_root: Path) -> Dict[str, SchemaJob]:
    jobs: Dict[str, SchemaJob] = {}
    for metadata_path in sorted(raw_root.glob("*/metadata.json")):
        metadata = load_metadata(metadata_path)
        schema = metadata["query"]["schema"]
        jobs[schema] = SchemaJob(schema=schema, directory=metadata_path.parent)
    return jobs


def discover_schema_jobs_for_symbol(raw_root: Path, symbol: str) -> Dict[str, SchemaJob]:
    jobs: Dict[str, SchemaJob] = {}
    for metadata_path in sorted(raw_root.glob("*/metadata.json")):
        metadata = load_metadata(metadata_path)
        query = metadata["query"]
        symbols = set(query.get("symbols", []))
        if symbol not in symbols:
            continue
        schema = query["schema"]
        if schema in jobs:
            raise RuntimeError(
                f"Found multiple raw directories for schema '{schema}' and symbol '{symbol}'. "
                f"Please keep only one matching job directory per schema."
            )
        jobs[schema] = SchemaJob(schema=schema, directory=metadata_path.parent)
    return jobs


def parse_session_time(value: str) -> time:
    return time.fromisoformat(value)


def localize_timestamp(ts_event: str, timezone_name: str) -> tuple[datetime, str]:
    # Preserve nanosecond precision by reusing the original fractional suffix.
    date_part, time_part = ts_event[:-1].split("T")
    if "." in time_part:
        hms_part, fractional = time_part.split(".", 1)
    else:
        hms_part, fractional = time_part, ""
    base = datetime.fromisoformat(f"{date_part}T{hms_part}+00:00")
    local_dt = base.astimezone(ZoneInfo(timezone_name)).replace(tzinfo=None)
    if fractional:
        return local_dt, f"{local_dt:%Y-%m-%d %H:%M:%S}.{fractional}"
    return local_dt, f"{local_dt:%Y-%m-%d %H:%M:%S}"


def should_keep_timestamp(local_dt: datetime, session_start: time, session_end: time) -> bool:
    return session_start <= local_dt.time() <= session_end


def price_to_int(price: str) -> int:
    if "." in price:
        whole, frac = price.split(".", 1)
    else:
        whole, frac = price, ""
    frac = (frac + "000000000")[:9]
    sign = -1 if whole.startswith("-") else 1
    whole_int = abs(int(whole))
    return sign * (whole_int * 1_000_000_000 + int(frac))


def int_to_price(price: int) -> str:
    sign = "-" if price < 0 else ""
    price = abs(price)
    whole = price // 1_000_000_000
    frac = price % 1_000_000_000
    return f"{sign}{whole}.{frac:09d}".rstrip("0").rstrip(".")


def stream_copy(src: TextIO, dst: TextIO, chunk_size: int = 1024 * 1024) -> None:
    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            break
        dst.write(chunk)


@contextmanager
def open_csv_text(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".zst":
        process = subprocess.Popen(
            ["zstdcat", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        try:
            yield process.stdout
        finally:
            process.stdout.close()
            process.wait()
            if process.returncode:
                stderr = process.stderr.read() if process.stderr else ""
                stderr_clean = stderr.strip()
                benign_early_close = (
                    "Broken pipe" in stderr
                    or stderr_clean in {"", "zstd:"}
                )
                if not benign_early_close:
                    raise RuntimeError(f"zstdcat failed for {path}: {stderr}")
            if process.stderr is not None:
                process.stderr.close()
    else:
        with path.open("r", newline="") as handle:
            yield handle


def csv_reader(path: Path) -> Iterator[dict[str, str]]:
    with open_csv_text(path) as handle:
        yield from csv.DictReader(handle)


def decompress_file(path: Path, *, delete_source: bool) -> Path:
    if path.suffix != ".zst":
        return path
    output_path = path.with_suffix("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_csv_text(path) as src, output_path.open("w") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    if delete_source:
        path.unlink()
    return output_path
