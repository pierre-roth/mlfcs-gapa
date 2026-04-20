from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from piroth.config import GenerateConfig, PretrainConfig, ReportConfig, SuiteConfig, SweepConfig, TrainConfig

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _resolve_runner(kind: str):
    if kind == "generate":
        from piroth.simulator import generate_dataset

        return GenerateConfig, generate_dataset
    if kind == "pretrain":
        from piroth.pretrain import run_pretrain

        return PretrainConfig, run_pretrain
    if kind == "train":
        from piroth.train import run_train

        return TrainConfig, run_train
    if kind == "report":
        from piroth.report import run_report

        return ReportConfig, run_report
    if kind == "suite":
        from piroth.run_suite import run_suite

        return SuiteConfig, run_suite
    if kind == "sweep":
        from piroth.sweep import run_sweep

        return SweepConfig, run_sweep
    raise KeyError(kind)


def _field_default(field) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()
    return None


def _strip_optional(field_type: Any) -> tuple[Any, bool]:
    origin = get_origin(field_type)
    args = get_args(field_type)
    if origin in {None, list}:
        return field_type, False
    non_none = [arg for arg in args if arg is not type(None)]
    if len(non_none) == 1 and len(non_none) != len(args):
        return non_none[0], True
    return field_type, False


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    raise ValueError(f"Cannot parse boolean value from {raw!r}")


def _parse_list(raw: str, item_type: Any) -> list[Any]:
    stripped = raw.strip()
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list, got {type(payload).__name__}")
        return [_coerce_value(item if isinstance(item, str) else str(item), item_type, None) for item in payload]
    items = [item.strip() for item in stripped.split(",") if item.strip()]
    return [_coerce_value(item, item_type, None) for item in items]


def _coerce_value(raw: str, field_type: Any, default: Any) -> Any:
    target_type, is_optional = _strip_optional(field_type)
    if is_optional and raw.strip().lower() in {"", "none", "null"}:
        return None
    origin = get_origin(target_type)
    if origin is list:
        item_type = get_args(target_type)[0] if get_args(target_type) else str
        return _parse_list(raw, item_type)
    if target_type is bool:
        return _parse_bool(raw)
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    if target_type is str:
        return raw
    if target_type is Any or target_type is None:
        if isinstance(default, bool):
            return _parse_bool(raw)
        if isinstance(default, int) and not isinstance(default, bool):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, list):
            item_type = type(default[0]) if default else str
            return _parse_list(raw, item_type)
        return raw
    if isinstance(target_type, type):
        return target_type(raw)
    return raw


def _load_overrides(config_cls, set_items: list[str]) -> dict[str, Any]:
    by_name = {field.name: field for field in fields(config_cls)}
    type_hints = get_type_hints(config_cls)
    overrides: dict[str, Any] = {}
    for name, field in by_name.items():
        env_value = os.environ.get(name.upper())
        if env_value is None:
            continue
        overrides[name] = _coerce_value(env_value, type_hints.get(name, field.type), _field_default(field))
    for item in set_items:
        if "=" not in item:
            raise SystemExit(f"Invalid --set value {item!r}; expected field=value")
        name, raw = item.split("=", 1)
        if name not in by_name:
            raise SystemExit(f"Unknown config field: {name}")
        field = by_name[name]
        overrides[name] = _coerce_value(raw, type_hints.get(name, field.type), _field_default(field))
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch Euler jobs into the piroth simulated continuous pipeline.")
    parser.add_argument("kind", choices=["generate", "pretrain", "train", "report", "suite", "sweep"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args()
    config_cls, runner = _resolve_runner(args.kind)
    overrides = _load_overrides(config_cls, args.set)
    config = config_cls(**overrides)
    config.apply_mode_defaults()
    if args.dry_run:
        payload = {field.name: getattr(config, field.name) for field in fields(config_cls)}
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    runner(config)


if __name__ == "__main__":
    main()
