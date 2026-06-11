import sys
from pathlib import Path
from types import SimpleNamespace

from mlfcs_gapa.experiments.tracking import (
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    init_wandb_run,
)


def test_wandb_tracking_disabled_does_not_import_wandb() -> None:
    tracker = init_wandb_run(
        enabled=False,
        job_type="unit",
        config={"path": Path("runs/example")},
    )

    assert not tracker.enabled
    tracker.log({"metric": 1.0})
    tracker.log_metrics({"metric": 1.0})
    tracker.finish()


def test_wandb_tracking_uses_project_defaults_and_logs_artifact(tmp_path, monkeypatch) -> None:
    artifact_path = tmp_path / "metrics.csv"
    artifact_path.write_text("metric,value\nf1,1.0\n", encoding="utf-8")
    fake_run = _FakeRun()
    def fake_init(**kwargs):
        fake_run.init_kwargs = kwargs
        return fake_run

    fake_wandb = SimpleNamespace(
        init=fake_init,
        log=fake_run.log,
        Artifact=_FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    tracker = init_wandb_run(
        enabled=True,
        job_type="unit",
        config={"output_dir": tmp_path, "steps": 5},
        tags=("synthetic",),
    )
    tracker.log_metrics({"f1": 0.5, "model": "Attn-LOB"}, prefix="eval")
    tracker.update_summary({"f1": 0.5, "model": "Attn-LOB"}, prefix="eval")
    tracker.log_artifact(artifact_path, name="unit artifact", artifact_type="metrics")
    tracker.finish()

    assert tracker.enabled
    assert fake_run.init_kwargs["entity"] == DEFAULT_WANDB_ENTITY
    assert fake_run.init_kwargs["project"] == DEFAULT_WANDB_PROJECT
    assert fake_run.init_kwargs["config"]["output_dir"] == str(tmp_path)
    assert fake_run.logged == [({"eval/f1": 0.5}, None)]
    assert fake_run.summary["eval/f1"] == 0.5
    assert fake_run.artifacts[0].name == "unit-artifact"
    assert fake_run.artifacts[0].files == [str(artifact_path)]
    assert fake_run.finished


class _FakeRun:
    def __init__(self) -> None:
        self.init_kwargs = {}
        self.logged = []
        self.summary = {}
        self.artifacts = []
        self.finished = False

    def log(self, values, step=None) -> None:
        self.logged.append((values, step))

    def log_artifact(self, artifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


class _FakeArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files = []
        self.dirs = []

    def add_file(self, path: str) -> None:
        self.files.append(path)

    def add_dir(self, path: str) -> None:
        self.dirs.append(path)
