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

from piroth.config import DiagnosticsConfig
from piroth.diagnostics import run_diagnostics
from piroth.paper_experiments import run_ablation_suite, run_full_paper_suite, run_latency_suite, run_paper_baseline_suite
from piroth.real_data import load_market_days
from piroth.synthetic_validation import run_synthetic_validation_suite
from piroth.training import evaluate_trained_policy, train_dqn, train_ppo, train_pretrain_classifier
from piroth.visualizer import build_synthetic_data_report, load_exported_days

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Euler dispatcher for the piroth2 simulator branch.")
    parser.add_argument(
        "kind",
        choices=[
            "diagnostics",
            "visualize",
            "paper-baselines",
            "synthetic-validation",
            "latency-suite",
            "ablation-suite",
            "paper-suite",
            "pretrain",
            "train-ppo",
            "train-dqn",
            "evaluate-ppo",
            "evaluate-dqn",
        ],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument("--export-root", default="")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args()
    config = DiagnosticsConfig(**_load_overrides(DiagnosticsConfig, args.set))
    config.apply_mode_defaults()
    if args.dry_run:
        print(json.dumps({field.name: getattr(config, field.name) for field in fields(DiagnosticsConfig)}, indent=2, sort_keys=True))
        return
    if args.kind == "diagnostics":
        run_diagnostics(config)
        return
    if args.kind == "paper-baselines":
        run_paper_baseline_suite(config)
        return
    if args.kind == "synthetic-validation":
        run_synthetic_validation_suite(
            config,
            seeds=_env_int_list("VALIDATION_SEEDS", [7, 11, 17, 23]),
            symbols=_env_str_list("VALIDATION_SYMBOLS", [config.symbol]),
            days_per_case=int(os.environ.get("VALIDATION_DAYS_PER_CASE", "2")),
            events_per_day=_env_optional_int("VALIDATION_EVENTS_PER_DAY", config.events_per_day_override or 12_000),
            report_cases=int(os.environ.get("VALIDATION_REPORT_CASES", "4")),
            export_report_days=_env_bool("VALIDATION_EXPORT_REPORT_DAYS", True),
        )
        return
    if args.kind == "latency-suite":
        run_latency_suite(config)
        return
    if args.kind == "ablation-suite":
        run_ablation_suite(config, device=args.device)
        return
    if args.kind == "paper-suite":
        run_full_paper_suite(config, device=args.device)
        return
    output_dir = config.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.kind == "visualize":
        if args.export_root:
            days = load_exported_days(Path(args.export_root), config.symbol)
        else:
            days = load_market_days(config, "test")[: min(config.export_day_count, config.test_days)]
        build_synthetic_data_report(days, config, output_dir / "visual_report")
    elif args.kind == "pretrain":
        days = load_market_days(config, "train")
        eval_days = load_market_days(config, "test")
        train_pretrain_classifier(days, config, output_dir / "models", device=args.device, eval_days=eval_days)
    elif args.kind == "train-ppo":
        days = load_market_days(config, "train")
        pretrain_path = Path(args.checkpoint) if args.checkpoint else output_dir / "models" / "attnlob_pretrain.pt"
        train_ppo(days, config, output_dir / "models", pretrain_path=pretrain_path, device=args.device)
    elif args.kind == "train-dqn":
        days = load_market_days(config, "train")
        pretrain_path = Path(args.checkpoint) if args.checkpoint else output_dir / "models" / "attnlob_pretrain.pt"
        train_dqn(days, config, output_dir / "models", pretrain_path=pretrain_path, device=args.device)
    elif args.kind in {"evaluate-ppo", "evaluate-dqn"}:
        days = load_market_days(config, "test")
        kind = "ppo" if args.kind == "evaluate-ppo" else "dqn"
        default_checkpoint = output_dir / "models" / ("c_ppo.pt" if kind == "ppo" else "d_dqn.pt")
        evaluate_trained_policy(days, config, Path(args.checkpoint) if args.checkpoint else default_checkpoint, kind, output_dir, device=args.device)


def _field_default(field) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()
    return None


def _load_overrides(config_cls, set_items: list[str]) -> dict[str, Any]:
    by_name = {field.name: field for field in fields(config_cls)}
    type_hints = get_type_hints(config_cls)
    overrides: dict[str, Any] = {}
    for name, field in by_name.items():
        env_value = os.environ.get(name.upper())
        if env_value is not None:
            overrides[name] = _coerce(env_value, type_hints.get(name, field.type), _field_default(field))
    for item in set_items:
        if "=" not in item:
            raise SystemExit(f"Invalid --set value {item!r}; expected field=value")
        key, raw = item.split("=", 1)
        if key not in by_name:
            raise SystemExit(f"Unknown config field: {key}")
        overrides[key] = _coerce(raw, type_hints.get(key, by_name[key].type), _field_default(by_name[key]))
    return overrides


def _coerce(raw: str, annotation: Any, default: Any) -> Any:
    target, is_optional = _strip_optional(annotation)
    if is_optional and raw.strip().lower() in {"", "none", "null"}:
        return None
    origin = get_origin(target)
    if origin is list:
        item_type = get_args(target)[0] if get_args(target) else str
        return [_coerce(item.strip(), item_type, None) for item in _split_list(raw)]
    if target is bool:
        return _parse_bool(raw)
    if target is int:
        return int(raw)
    if target is float:
        return float(raw)
    if target is str:
        return raw
    if isinstance(default, bool):
        return _parse_bool(raw)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        item_type = type(default[0]) if default else str
        return [_coerce(item.strip(), item_type, None) for item in _split_list(raw)]
    return raw


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    non_none = [arg for arg in args if arg is not type(None)]
    if non_none and len(non_none) != len(args):
        return non_none[0], True
    return annotation, False


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    raise ValueError(f"Cannot parse boolean value from {raw!r}")


def _split_list(raw: str) -> list[str]:
    stripped = raw.strip()
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list, got {type(payload).__name__}")
        return [str(item) for item in payload]
    return [item.strip() for item in stripped.split(",") if item.strip()]


def _env_str_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _split_list(raw)


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return [int(item) for item in _split_list(raw)]


def _env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.strip().lower() in {"", "none", "null"}:
        return None
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _parse_bool(raw)


if __name__ == "__main__":
    main()
