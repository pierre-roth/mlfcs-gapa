from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import shutil
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from piroth.config import DEFAULT_SYMBOLS, DiagnosticsConfig
from piroth.data_quality import assess_synthetic_quality
from piroth.simulator import SyntheticMarketGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a persistent piroth2 synthetic month dataset.")
    parser.add_argument("--dataset-name", default=os.environ.get("DATASET_NAME", "synthetic_month_v1"))
    parser.add_argument("--output-root", default=os.environ.get("SYNTHETIC_DATA_ROOT", f"/cluster/work/math/{os.environ.get('USER', 'piroth')}/mlfcs-gapa/data/persistent_synthetic"))
    parser.add_argument("--symbols", nargs="+", default=_split_env("SYMBOLS", ["000001", "000858", "002415"]))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "7")))
    parser.add_argument("--num-days", type=int, default=int(os.environ.get("NUM_DAYS", "22")))
    parser.add_argument("--events-per-day", type=int, default=_env_int("EVENTS_PER_DAY_OVERRIDE", 0), help="0 keeps each symbol's configured event count.")
    parser.add_argument("--quality-days", type=int, default=int(os.environ.get("QUALITY_DAYS", "4")), help="Number of generated days per symbol retained for quality summary.")
    parser.add_argument("--checksums", action="store_true", default=_env_bool("CHECKSUMS", False), help="Hash every CSV file after generation. Disabled by default to reduce IO.")
    parser.add_argument("--overwrite", action="store_true", default=_env_bool("OVERWRITE", False))
    args = parser.parse_args()

    root = Path(args.output_root) / args.dataset_name
    tmp_root = root.with_name(f".{root.name}.tmp-{int(time.time())}")
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"Dataset already exists: {root} (set OVERWRITE=1 to replace)")
        shutil.rmtree(root)
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    started = time.time()
    rows: list[dict[str, object]] = []
    quality: dict[str, object] = {}
    config_records: dict[str, object] = {}
    for symbol in args.symbols:
        if symbol not in DEFAULT_SYMBOLS:
            raise KeyError(f"Unsupported synthetic symbol {symbol!r}")
        config = DiagnosticsConfig(
            mode="full",
            data_source="synthetic",
            symbol=symbol,
            seed=args.seed,
            num_days=args.num_days,
            train_days=args.num_days,
            test_days=0,
            events_per_day_override=args.events_per_day or None,
            synthetic_build_depth_cube=False,
            order_flow_memory=float(os.environ.get("ORDER_FLOW_MEMORY", "0.35")),
            volatility_cluster_strength=float(os.environ.get("VOLATILITY_CLUSTER_STRENGTH", "0.45")),
            volatility_cluster_persistence=float(os.environ.get("VOLATILITY_CLUSTER_PERSISTENCE", "0.992")),
            run_name=args.dataset_name,
        )
        generator = SyntheticMarketGenerator(config)
        sample_days = []
        symbol_started = time.time()
        for day_idx, day_name in enumerate(generator.business_days(), start=1):
            day_started = time.time()
            day = generator.generate_day(day_name)
            day.export(tmp_root)
            if len(sample_days) < args.quality_days:
                sample_days.append(day)
            rows.append(
                {
                    "symbol": symbol,
                    "day": day_name,
                    "events": int(len(day.price)),
                    "trades": int(len(day.trades)),
                    "elapsed_sec": round(time.time() - day_started, 3),
                }
            )
            print(f"[{symbol}] exported {day_idx}/{args.num_days} {day_name}: events={len(day.price)} trades={len(day.trades)}", flush=True)
        quality[symbol] = assess_synthetic_quality(sample_days, config)
        config_records[symbol] = asdict(config)
        print(f"[{symbol}] completed in {time.time() - symbol_started:.1f}s", flush=True)

    metadata = {
        "dataset_name": args.dataset_name,
        "root": str(root),
        "tmp_root": str(tmp_root),
        "seed": args.seed,
        "symbols": args.symbols,
        "num_days": args.num_days,
        "events_per_day_override": args.events_per_day or None,
        "generator": "piroth.SyntheticMarketGenerator",
        "generator_options": {
            "order_flow_memory": float(os.environ.get("ORDER_FLOW_MEMORY", "0.35")),
            "volatility_cluster_strength": float(os.environ.get("VOLATILITY_CLUSTER_STRENGTH", "0.45")),
            "volatility_cluster_persistence": float(os.environ.get("VOLATILITY_CLUSTER_PERSISTENCE", "0.992")),
            "synthetic_build_depth_cube": False,
        },
        "created_at_unix": int(started),
        "elapsed_sec": round(time.time() - started, 3),
        "days": rows,
        "quality_sample_days": args.quality_days,
        "quality": quality,
        "configs": config_records,
    }
    _write_json(tmp_root / "metadata.json", metadata)
    if args.checksums:
        _write_json(tmp_root / "checksums.json", _checksums(tmp_root))
    tmp_root.rename(root)
    print(f"Persistent synthetic dataset written to {root}", flush=True)


def _split_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    return raw.split() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in {None, ""} else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _checksums(root: Path) -> dict[str, dict[str, object]]:
    checksums = {}
    for path in sorted(root.rglob("*.csv")):
        checksums[str(path.relative_to(root))] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return checksums


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
