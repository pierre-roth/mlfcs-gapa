from __future__ import annotations

import argparse
from pathlib import Path

from .analyze import DivergenceConfig, analyze_symbol
from .common import decompress_file, discover_all_schema_jobs
from .convert import ConversionConfig, convert_symbol
from .progress import DatabentoProgress
from .validate import ValidationConfig, validate_symbol


def decompress_raw(raw_root: Path, delete_source: bool) -> None:
    jobs = discover_all_schema_jobs(raw_root)
    paths = []
    for job in jobs:
        for path in sorted(job.directory.glob("*.zst")):
            paths.append(path)

    with DatabentoProgress() as progress:
        overall_task = progress.add_task("[cyan]Decompressing raw Databento files[/cyan]", total=len(paths))
        for path in paths:
            task_id = progress.add_task(f"[magenta]Decompressing[/magenta] {path.name}", total=None)
            output_path = path.with_suffix("")
            if output_path.exists():
                if delete_source:
                    path.unlink()
                progress.update(task_id, description=f"[green]Skipped existing[/green] {output_path.name}", completed=1, total=1)
            else:
                decompress_file(path, delete_source=delete_source)
                progress.update(task_id, description=f"[green]Decompressed[/green] {output_path.name}", completed=1, total=1)
            progress.update(overall_task, advance=1)
            progress.remove_task(task_id)
        progress.print_summary("[bold green]Decompression complete.[/bold green]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Databento preprocessing tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    decompress_parser = subparsers.add_parser("decompress", help="Decompress .zst files in data/raw")
    decompress_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    decompress_parser.add_argument("--delete-source", action="store_true")

    convert_parser = subparsers.add_parser("convert", help="Convert Databento raw data into project CSVs")
    convert_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    convert_parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    convert_parser.add_argument("--symbol", default="GOOGL")
    convert_parser.add_argument("--timezone", default="America/New_York")
    convert_parser.add_argument("--session-start", default="09:30:00")
    convert_parser.add_argument("--session-end", default="16:00:00")
    convert_parser.add_argument("--day", action="append", help="Restrict conversion to one or more YYYYMMDD days")

    validate_parser = subparsers.add_parser("validate", help="Validate processed outputs against Databento MBP-10")
    validate_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    validate_parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    validate_parser.add_argument("--validation-root", type=Path, default=Path("data/validation"))
    validate_parser.add_argument("--symbol", default="GOOGL")
    validate_parser.add_argument("--timezone", default="America/New_York")
    validate_parser.add_argument("--session-start", default="09:30:00")
    validate_parser.add_argument("--session-end", default="16:00:00")
    validate_parser.add_argument("--sample-every", type=int, default=10_000)
    validate_parser.add_argument("--edge-samples", type=int, default=1_000)
    validate_parser.add_argument("--day", action="append", help="Restrict validation to one or more YYYYMMDD days")

    analyze_parser = subparsers.add_parser("analyze-divergence", help="Analyze why processed book snapshots diverge from MBP-10")
    analyze_parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    analyze_parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    analyze_parser.add_argument("--analysis-root", type=Path, default=Path("data/validation"))
    analyze_parser.add_argument("--symbol", default="GOOGL")
    analyze_parser.add_argument("--timezone", default="America/New_York")
    analyze_parser.add_argument("--session-start", default="09:30:00")
    analyze_parser.add_argument("--session-end", default="16:00:00")
    analyze_parser.add_argument("--sample-every", type=int, default=10_000)
    analyze_parser.add_argument("--edge-samples", type=int, default=1_000)
    analyze_parser.add_argument("--tolerance-us", type=int, default=1_000)
    analyze_parser.add_argument("--examples", type=int, default=10)
    analyze_parser.add_argument("--day", action="append", help="Restrict analysis to one or more YYYYMMDD days")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "decompress":
        decompress_raw(args.raw_root, delete_source=args.delete_source)
        return

    if args.command == "convert":
        with DatabentoProgress() as progress:
            results = convert_symbol(
                ConversionConfig(
                    raw_root=args.raw_root,
                    output_root=args.output_root,
                    symbol=args.symbol,
                    timezone_name=args.timezone,
                    session_start=args.session_start,
                    session_end=args.session_end,
                    sample_every=args.sample_every,
                    edge_samples=args.edge_samples,
                    days=set(args.day) if args.day else None,
                ),
                progress=progress,
            )
            progress.print_summary(
                f"[bold cyan]Finished {len(results)} day(s) for {args.symbol}[/bold cyan]"
            )
        return

    if args.command == "validate":
        with DatabentoProgress() as progress:
            results = validate_symbol(
                ValidationConfig(
                    raw_root=args.raw_root,
                    processed_root=args.processed_root,
                    validation_root=args.validation_root,
                    symbol=args.symbol,
                    timezone_name=args.timezone,
                    session_start=args.session_start,
                    session_end=args.session_end,
                    days=set(args.day) if args.day else None,
                ),
                progress=progress,
            )
            progress.print_summary(
                f"[bold cyan]Validated {len(results)} day(s) for {args.symbol}[/bold cyan]"
            )
        return

    if args.command == "analyze-divergence":
        with DatabentoProgress() as progress:
            results = analyze_symbol(
                DivergenceConfig(
                    raw_root=args.raw_root,
                    processed_root=args.processed_root,
                    analysis_root=args.analysis_root,
                    symbol=args.symbol,
                    timezone_name=args.timezone,
                    session_start=args.session_start,
                    session_end=args.session_end,
                    sample_every=args.sample_every,
                    edge_samples=args.edge_samples,
                    tolerance_us=args.tolerance_us,
                    examples=args.examples,
                    days=set(args.day) if args.day else None,
                ),
                progress=progress,
            )
            progress.print_summary(
                f"[bold cyan]Analyzed divergence for {len(results)} day(s) of {args.symbol}[/bold cyan]"
            )
        return

    parser.error(f"Unknown command {args.command}")
