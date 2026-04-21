from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, get_args, get_origin

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from piroth.config import DiagnosticsConfig
from piroth.diagnostics import run_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Euler dispatcher for the piroth2 simulator branch.")
    parser.add_argument("kind", choices=["diagnostics"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args()
    config = DiagnosticsConfig(**_load_overrides(DiagnosticsConfig, args.set))
    config.apply_mode_defaults()
    if args.dry_run:
        print(json.dumps({field.name: getattr(config, field.name) for field in fields(DiagnosticsConfig)}, indent=2, sort_keys=True))
        return
    if args.kind == "diagnostics":
        run_diagnostics(config)


def _field_default(field) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()
    return None


def _load_overrides(config_cls, set_items: list[str]) -> dict[str, Any]:
    by_name = {field.name: field for field in fields(config_cls)}
    overrides: dict[str, Any] = {}
    for name, field in by_name.items():
        env_value = os.environ.get(name.upper())
        if env_value is not None:
            overrides[name] = _coerce(env_value, field.type, _field_default(field))
    for item in set_items:
        key, raw = item.split("=", 1)
        if key not in by_name:
            raise SystemExit(f"Unknown config field: {key}")
        overrides[key] = _coerce(raw, by_name[key].type, _field_default(by_name[key]))
    return overrides


def _coerce(raw: str, annotation: Any, default: Any) -> Any:
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
    if args and type(None) in args:
        inner = next(arg for arg in args if arg is not type(None))
        return None if raw.strip().lower() in {"none", "null", ""} else _coerce(raw, inner, default)
    if isinstance(default, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


if __name__ == "__main__":
    main()
