from __future__ import annotations

from pathlib import Path

from dataclasses import asdict, fields

import pyrallis

from .config import GenerateConfig, PretrainConfig, ReportConfig, SuiteConfig, TrainConfig
from .pretrain import run_pretrain
from .report import run_report
from .simulator import generate_dataset
from .train import run_train
from .utils import save_json


def _cast(config: SuiteConfig, target_type):
    payload = asdict(config)
    allowed = {field.name for field in fields(target_type)}
    return target_type(**{key: value for key, value in payload.items() if key in allowed})


def run_suite(config: SuiteConfig) -> dict[str, object]:
    config.apply_mode_defaults()
    summary: dict[str, object] = {}
    if config.generate_data:
        summary["generate"] = generate_dataset(_cast(config, GenerateConfig))
    if config.run_pretrain:
        summary["pretrain"] = run_pretrain(_cast(config, PretrainConfig))
    if config.run_train:
        summary["train"] = run_train(_cast(config, TrainConfig))
    if config.run_report:
        summary["report"] = run_report(_cast(config, ReportConfig))
    save_json(Path(config.output_dir()) / "suite_summary.json", summary)
    return summary


@pyrallis.wrap()
def main(config: SuiteConfig) -> None:
    run_suite(config)


if __name__ == "__main__":
    main()
