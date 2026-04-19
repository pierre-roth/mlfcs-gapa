from __future__ import annotations

import argparse
import inspect
from dataclasses import MISSING, fields, is_dataclass
from functools import wraps
from typing import Any, get_args, get_origin, get_type_hints

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _field_default(field) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()
    return None


def _strip_optional(field_type: Any) -> tuple[Any, bool]:
    origin = get_origin(field_type)
    args = get_args(field_type)
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
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {raw!r}")


def _coerce(raw: str, field_type: Any, default: Any) -> Any:
    target_type, is_optional = _strip_optional(field_type)
    if is_optional and raw.strip().lower() in {"", "none", "null"}:
        return None
    origin = get_origin(target_type)
    if origin is list:
        item_type = get_args(target_type)[0] if get_args(target_type) else str
        return [_coerce(item.strip(), item_type, None) for item in raw.split(",") if item.strip()]
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
            return [_coerce(item.strip(), item_type, None) for item in raw.split(",") if item.strip()]
        return raw
    if isinstance(target_type, type):
        return target_type(raw)
    return raw


def wrap():
    def decorator(fn):
        signature = inspect.signature(fn)
        parameters = list(signature.parameters.values())
        if len(parameters) != 1:
            raise TypeError("pyrallis.wrap() expects a function with exactly one config argument")
        config_name = parameters[0].name
        type_hints = get_type_hints(fn)
        config_type = type_hints.get(config_name, parameters[0].annotation)
        if not is_dataclass(config_type):
            raise TypeError("pyrallis.wrap() expects the config argument to be a dataclass type")

        @wraps(fn)
        def wrapped(*args, **kwargs):
            if args or kwargs:
                return fn(*args, **kwargs)
            parser = argparse.ArgumentParser()
            config_hints = get_type_hints(config_type)
            for field in fields(config_type):
                parser.add_argument(f"--{field.name}", dest=field.name)
            parsed = parser.parse_args()
            values = {}
            for field in fields(config_type):
                raw = getattr(parsed, field.name)
                if raw is None:
                    continue
                values[field.name] = _coerce(raw, config_hints.get(field.name, field.type), _field_default(field))
            return fn(config_type(**values))

        return wrapped

    return decorator
