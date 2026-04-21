from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any, get_args, get_origin

from .config import DiagnosticsConfig
from .diagnostics import run_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulator-only diagnostics for the piroth2 branch.")
    parser.add_argument("--mode", default="medium")
    parser.add_argument("--symbol", default="000001")
    parser.add_argument("--run-name", default="piroth2_diagnostics")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args()

    config = DiagnosticsConfig(mode=args.mode, symbol=args.symbol, run_name=args.run_name, seed=args.seed)
    overrides = _parse_overrides(args.set)
    for key, value in overrides.items():
        setattr(config, key, value)
    run_diagnostics(config)


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    config_fields = {field.name: field for field in fields(DiagnosticsConfig)}
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid override {item!r}; expected field=value")
        key, raw = item.split("=", 1)
        if key not in config_fields:
            raise SystemExit(f"Unknown config field: {key}")
        parsed[key] = _coerce(raw, config_fields[key].type)
    return parsed


def _coerce(raw: str, annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list:
        return [item.strip() for item in raw.split(",") if item.strip()]
    if annotation is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    return raw


if __name__ == "__main__":
    main()
