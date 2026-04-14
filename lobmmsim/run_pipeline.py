from __future__ import annotations

from dataclasses import asdict, fields

import pyrallis

from .config import GenerateConfig, PretrainConfig, RLTrainConfig, SuiteConfig
from .pretrain import run_pretrain
from .report import run_report
from .simulator import generate_dataset
from .train_rl import run_rl_training


def _cast_config(config: SuiteConfig, target_type):
    source = asdict(config)
    allowed = {field.name for field in fields(target_type)}
    return target_type(**{key: value for key, value in source.items() if key in allowed})


def run_suite(config: SuiteConfig) -> None:
    config.apply_mode_defaults()
    if config.generate_data:
        generate_dataset(_cast_config(config, GenerateConfig))
    if config.run_pretrain:
        run_pretrain(_cast_config(config, PretrainConfig))
    if config.run_rl:
        run_rl_training(_cast_config(config, RLTrainConfig))
    if config.run_report:
        run_report(_cast_config(config, RLTrainConfig))


@pyrallis.wrap()
def main(config: SuiteConfig) -> None:
    run_suite(config)


if __name__ == "__main__":
    main()
