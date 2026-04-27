from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any, get_args, get_origin, get_type_hints

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
    type_hints = get_type_hints(DiagnosticsConfig)
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid override {item!r}; expected field=value")
        key, raw = item.split("=", 1)
        if key not in config_fields:
            raise SystemExit(f"Unknown config field: {key}")
        parsed[key] = _coerce(raw, type_hints.get(key, config_fields[key].type), getattr(DiagnosticsConfig(), key))
    return parsed


def _coerce(raw: str, annotation: Any, default: Any) -> Any:
    target, is_optional = _strip_optional(annotation)
    if is_optional and raw.strip().lower() in {"", "none", "null"}:
        return None
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    origin = get_origin(target)
    if origin is list:
        return [item.strip() for item in raw.split(",") if item.strip()]
    if target is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if target is int:
        return int(raw)
    if target is float:
        return float(raw)
    return raw


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    args = get_args(annotation)
    non_none = [arg for arg in args if arg is not type(None)]
    if non_none and len(non_none) != len(args):
        return non_none[0], True
    return annotation, False


if __name__ == "__main__":
    main()
