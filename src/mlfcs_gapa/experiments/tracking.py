"""Optional Weights & Biases tracking for experiment entrypoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

DEFAULT_WANDB_ENTITY = "piroth-ethz"
DEFAULT_WANDB_PROJECT = "mm-drl-lob"


@dataclass
class WandbTracker:
    """Small wrapper that keeps W&B optional outside experiment commands."""

    run: Any | None = None
    module: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.run is not None and self.module is not None

    def log(self, values: Mapping[str, Any], *, step: int | None = None) -> None:
        if not self.enabled:
            return
        self.module.log(_clean_mapping(values), step=step)

    def log_metrics(
        self, values: Mapping[str, Any], *, prefix: str | None = None, step: int | None = None
    ) -> None:
        if not self.enabled:
            return
        numeric = _numeric_mapping(values, prefix=prefix)
        if numeric:
            self.module.log(numeric, step=step)

    def update_summary(self, values: Mapping[str, Any], *, prefix: str | None = None) -> None:
        if not self.enabled:
            return
        for key, value in _numeric_mapping(values, prefix=prefix).items():
            self.run.summary[key] = value

    def log_artifact(
        self,
        paths: Path | Sequence[Path],
        *,
        name: str,
        artifact_type: str,
    ) -> None:
        if not self.enabled:
            return
        artifact = self.module.Artifact(_safe_artifact_name(name), type=artifact_type)
        for path in _as_paths(paths):
            if not path.exists():
                continue
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.enabled:
            self.run.finish()


@contextmanager
def wandb_run(
    *,
    enabled: bool,
    job_type: str,
    config: Mapping[str, Any],
    entity: str = DEFAULT_WANDB_ENTITY,
    project: str = DEFAULT_WANDB_PROJECT,
    mode: str | None = None,
    name: str | None = None,
    group: str | None = None,
    tags: Sequence[str] = (),
) -> Any:
    tracker = init_wandb_run(
        enabled=enabled,
        job_type=job_type,
        config=config,
        entity=entity,
        project=project,
        mode=mode,
        name=name,
        group=group,
        tags=tags,
    )
    try:
        yield tracker
    finally:
        tracker.finish()


def init_wandb_run(
    *,
    enabled: bool,
    job_type: str,
    config: Mapping[str, Any],
    entity: str = DEFAULT_WANDB_ENTITY,
    project: str = DEFAULT_WANDB_PROJECT,
    mode: str | None = None,
    name: str | None = None,
    group: str | None = None,
    tags: Sequence[str] = (),
) -> WandbTracker:
    if not enabled or mode == "disabled":
        return WandbTracker()

    wandb = import_module("wandb")
    init_kwargs: dict[str, Any] = {
        "entity": entity,
        "project": project,
        "job_type": job_type,
        "config": _clean_mapping(config),
        "tags": list(tags),
    }
    if mode:
        init_kwargs["mode"] = mode
    if name:
        init_kwargs["name"] = name
    if group:
        init_kwargs["group"] = group
    run = wandb.init(**init_kwargs)
    return WandbTracker(run=run, module=wandb)


def _as_paths(paths: Path | Sequence[Path]) -> list[Path]:
    if isinstance(paths, Path):
        return [paths]
    return [Path(path) for path in paths]


def _clean_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _clean_value(value) for key, value in values.items()}


def _clean_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_clean_value(item) for item in value]
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, Mapping):
        return _clean_mapping(value)
    return value


def _numeric_mapping(values: Mapping[str, Any], *, prefix: str | None = None) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            numeric_value = float(value)
        elif isinstance(value, int | float):
            numeric_value = float(value)
        else:
            continue
        metric_key = str(key)
        if prefix:
            metric_key = f"{prefix}/{metric_key}"
        cleaned[metric_key] = numeric_value
    return cleaned


def _safe_artifact_name(name: str) -> str:
    safe = []
    for character in name:
        if character.isalnum() or character in ("-", "_", "."):
            safe.append(character)
        else:
            safe.append("-")
    return "".join(safe).strip("-") or "artifact"
