from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


class DatabentoProgress:
    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )

    def __enter__(self) -> "DatabentoProgress":
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.progress.__exit__(exc_type, exc, tb)

    def add_task(self, description: str, total: Optional[int] = None) -> TaskID:
        return self.progress.add_task(description, total=total)

    def update(self, task_id: TaskID, *, advance: float = 0, description: Optional[str] = None, total: Optional[int] = None, completed: Optional[float] = None) -> None:
        kwargs = {}
        if description is not None:
            kwargs["description"] = description
        if total is not None:
            kwargs["total"] = total
        if completed is not None:
            kwargs["completed"] = completed
        self.progress.update(task_id, advance=advance, **kwargs)

    def remove_task(self, task_id: TaskID) -> None:
        self.progress.remove_task(task_id)

    def print_summary(self, message: str) -> None:
        self.console.print(message)

    @staticmethod
    def short_name(path: Path) -> str:
        return path.name
