from __future__ import annotations

import pyrallis

from .config import GenerateConfig
from .simulator import generate_dataset


@pyrallis.wrap()
def main(config: GenerateConfig) -> None:
    generate_dataset(config)


if __name__ == "__main__":
    main()
