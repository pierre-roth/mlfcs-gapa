from __future__ import annotations

import json
import signal
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import pyrallis
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import PretrainConfig
from .data import PretrainDataset
from .models import PretrainClassifier, build_backbone
from .pipeline import load_symbol_splits, prepare_run
from .utils import ensure_dir, save_json


class _StopRequested:
    def __init__(self) -> None:
        self.requested = False

    def handler(self, signum, frame) -> None:  # type: ignore[override]
        self.requested = True


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _symbol_paths(symbol_dir: Path, config: PretrainConfig) -> tuple[Path, Path, Path, Path]:
    final_path = symbol_dir / config.save_backbone_name
    checkpoint_path = symbol_dir / "checkpoint.pt"
    history_path = symbol_dir / "history.csv"
    summary_path = symbol_dir / "summary.json"
    return final_path, checkpoint_path, history_path, summary_path


def _dataset_sample_cap(config: PretrainConfig, split_name: str) -> int | None:
    if split_name == "train":
        return config.max_pretrain_samples_per_day
    return config.pretrain_eval_samples_per_day


def _dataset_class_weights(dataset: PretrainDataset, power: float) -> torch.Tensor:
    counts = dataset.class_counts()
    raw = torch.tensor([counts[label] for label in range(3)], dtype=torch.float32)
    safe = torch.clamp(raw, min=1.0)
    weights = (safe.sum() / safe).pow(power)
    return weights / weights.mean()


def _dataset_sampler(dataset: PretrainDataset, power: float) -> WeightedRandomSampler | None:
    if len(dataset) == 0:
        return None
    class_weights = _dataset_class_weights(dataset, power)
    sample_weights = class_weights[torch.from_numpy(dataset.sample_labels_np)]
    return WeightedRandomSampler(sample_weights.double(), num_samples=len(dataset), replacement=True)


def _loader_kwargs(config: PretrainConfig, pin_memory: bool) -> dict[str, object]:
    loader_kwargs: dict[str, object] = {
        "batch_size": config.pretrain_batch_size,
        "pin_memory": pin_memory,
        "num_workers": config.pretrain_num_workers,
    }
    if config.pretrain_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if config.pretrain_prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = config.pretrain_prefetch_factor
    return loader_kwargs


def _evaluate_classifier(
    model: PretrainClassifier,
    loader: DataLoader,
    device: str,
) -> dict[str, float | int | list[int]]:
    preds: list[int] = []
    targets: list[int] = []
    model.eval()
    with torch.no_grad():
        for lob, label in loader:
            logits = model(lob.to(device))
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(label.tolist())
    if not targets:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "target_class_counts": [0, 0, 0],
            "predicted_class_counts": [0, 0, 0],
            "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "samples": 0,
        }
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
    target_counts = [int(value) for value in np.bincount(np.asarray(targets, dtype=np.int64), minlength=3)]
    pred_counts = [int(value) for value in np.bincount(np.asarray(preds, dtype=np.int64), minlength=3)]
    matrix = confusion_matrix(targets, preds, labels=[0, 1, 2]).astype(int).tolist()
    accuracy = float(sum(int(p == t) for p, t in zip(preds, targets)) / len(targets))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": accuracy,
        "target_class_counts": target_counts,
        "predicted_class_counts": pred_counts,
        "confusion_matrix": matrix,
        "samples": int(len(targets)),
    }


def _save_checkpoint(
    *,
    symbol_dir: Path,
    config: PretrainConfig,
    model: PretrainClassifier,
    optimizer: Adam,
    history: list[dict[str, float]],
    best_f1: float,
    best_state: dict[str, torch.Tensor] | None,
    next_epoch: int,
    status: str,
    symbol: str,
) -> dict[str, float | str | int | bool | None]:
    final_path, checkpoint_path, history_path, summary_path = _symbol_paths(symbol_dir, config)
    latest_state = _cpu_state_dict(model.backbone)
    torch.save(latest_state, final_path)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "best_f1": best_f1,
        "best_state": best_state,
        "next_epoch": next_epoch,
        "status": status,
        "symbol": symbol,
        "final_backbone_path": str(final_path),
    }
    torch.save(payload, checkpoint_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    summary = {
        "symbol": symbol,
        "backbone_name": config.save_backbone_name,
        "backbone": config.pretrain_backbone,
        "best_f1": None if best_f1 < 0.0 else float(best_f1),
        "path": str(final_path),
        "checkpoint_path": str(checkpoint_path),
        "status": status,
        "epochs_completed": int(next_epoch),
        "pretrain_epochs_target": int(config.pretrain_epochs),
        "resume_enabled": bool(config.pretrain_resume),
        "pretrain_balance_mode": config.pretrain_balance_mode,
    }
    save_json(summary_path, summary)
    return summary


def run_pretrain(config: PretrainConfig) -> dict[str, dict[str, float | str]]:
    config.apply_mode_defaults()
    out_dir = prepare_run(config, label="pretrain")
    results: dict[str, dict[str, float | str]] = {}
    stop_requested = _StopRequested()
    old_usr1 = signal.getsignal(signal.SIGUSR1)
    old_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGUSR1, stop_requested.handler)
    signal.signal(signal.SIGTERM, stop_requested.handler)
    try:
        for symbol in config.symbols:
            symbol_dir = ensure_dir(out_dir / symbol / "pretrain")
            final_path, checkpoint_path, history_path, summary_path = _symbol_paths(symbol_dir, config)
            if config.pretrain_resume and checkpoint_path.exists():
                summary = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(summary, dict) and summary.get("status") == "completed" and final_path.exists():
                    loaded_summary = {
                        "symbol": symbol,
                        "backbone_name": config.save_backbone_name,
                        "backbone": config.pretrain_backbone,
                        "best_f1": float(summary.get("best_f1", -1.0)),
                        "path": str(final_path),
                        "checkpoint_path": str(checkpoint_path),
                        "status": "completed",
                        "epochs_completed": int(summary.get("next_epoch", config.pretrain_epochs)),
                        "pretrain_epochs_target": int(config.pretrain_epochs),
                        "resume_enabled": bool(config.pretrain_resume),
                        "pretrain_balance_mode": config.pretrain_balance_mode,
                    }
                    if summary_path.exists():
                        loaded_summary = {**loaded_summary, **json.loads(summary_path.read_text())}
                    save_json(summary_path, loaded_summary)
                    results[symbol] = loaded_summary
                    continue

            splits = load_symbol_splits(config, symbol)
            train_ds = PretrainDataset(
                splits["train"],
                lookback=config.lookback,
                horizon=config.pretrain_horizon,
                alpha=config.pretrain_alpha,
                max_samples_per_day=_dataset_sample_cap(config, "train"),
            )
            val_ds = PretrainDataset(
                splits["val"],
                lookback=config.lookback,
                horizon=config.pretrain_horizon,
                alpha=config.pretrain_alpha,
                max_samples_per_day=_dataset_sample_cap(config, "val"),
            )
            test_ds = PretrainDataset(
                splits["test"],
                lookback=config.lookback,
                horizon=config.pretrain_horizon,
                alpha=config.pretrain_alpha,
                max_samples_per_day=_dataset_sample_cap(config, "test"),
            )
            pin_memory = config.device == "cuda"
            loader_kwargs = _loader_kwargs(config, pin_memory)
            balance_mode = config.pretrain_balance_mode.strip().lower()
            use_weighted_loss = balance_mode in {"weighted_loss", "balanced_sampler_and_loss"}
            use_balanced_sampler = balance_mode in {"balanced_sampler", "balanced_sampler_and_loss"}
            sampler = _dataset_sampler(train_ds, config.pretrain_sampler_power) if use_balanced_sampler else None
            train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
            eval_loader_kwargs = dict(loader_kwargs)
            val_loader = DataLoader(val_ds, shuffle=False, **eval_loader_kwargs)
            test_loader = DataLoader(test_ds, shuffle=False, **eval_loader_kwargs)
            backbone = build_backbone(config.pretrain_backbone, config.lookback).to(config.device)
            model = PretrainClassifier(backbone).to(config.device)
            optimizer = Adam(model.parameters(), lr=config.pretrain_lr)
            class_weights_cpu = _dataset_class_weights(train_ds, config.pretrain_sampler_power)
            class_weights = class_weights_cpu.to(config.device) if use_weighted_loss else None
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            history: list[dict[str, float]] = []
            best_f1 = -1.0
            best_state: dict[str, torch.Tensor] | None = None
            start_epoch = 0
            if config.pretrain_resume and checkpoint_path.exists():
                payload = torch.load(checkpoint_path, map_location=config.device)
                model.load_state_dict(payload["model_state"])
                optimizer.load_state_dict(payload["optimizer_state"])
                history = list(payload.get("history", []))
                best_f1 = float(payload.get("best_f1", -1.0))
                best_state = payload.get("best_state")
                start_epoch = int(payload.get("next_epoch", 0))

            interrupted = False
            last_checkpoint = monotonic()
            for epoch in range(start_epoch, config.pretrain_epochs):
                model.train()
                train_loss = 0.0
                for lob, label in train_loader:
                    lob = lob.to(config.device)
                    label = label.to(config.device)
                    optimizer.zero_grad()
                    logits = model(lob)
                    loss = criterion(logits, label)
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss.item()) * lob.size(0)
                    now = monotonic()
                    if now - last_checkpoint >= config.pretrain_checkpoint_seconds:
                        _save_checkpoint(
                            symbol_dir=symbol_dir,
                            config=config,
                            model=model,
                            optimizer=optimizer,
                            history=history,
                            best_f1=best_f1,
                            best_state=best_state,
                            next_epoch=epoch,
                            status="running_partial",
                            symbol=symbol,
                        )
                        last_checkpoint = now
                    if stop_requested.requested:
                        interrupted = True
                        break
                if interrupted:
                    break
                val_metrics = _evaluate_classifier(model, val_loader, config.device)
                f1 = float(val_metrics["f1"])
                predicted_counts = list(val_metrics["predicted_class_counts"])  # type: ignore[arg-type]
                target_counts = list(val_metrics["target_class_counts"])  # type: ignore[arg-type]
                epoch_record = {
                    "epoch": epoch,
                    "train_loss": train_loss / max(len(train_ds), 1),
                    "precision": float(val_metrics["precision"]),
                    "recall": float(val_metrics["recall"]),
                    "f1": f1,
                    "accuracy": float(val_metrics["accuracy"]),
                    "target_count_0": float(target_counts[0]),
                    "target_count_1": float(target_counts[1]),
                    "target_count_2": float(target_counts[2]),
                    "pred_count_0": float(predicted_counts[0]),
                    "pred_count_1": float(predicted_counts[1]),
                    "pred_count_2": float(predicted_counts[2]),
                }
                history.append(epoch_record)
                if f1 > best_f1:
                    best_f1 = float(f1)
                    best_state = {key: value.cpu() for key, value in backbone.state_dict().items()}
                _save_checkpoint(
                    symbol_dir=symbol_dir,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    history=history,
                    best_f1=best_f1,
                    best_state=best_state,
                    next_epoch=epoch + 1,
                    status="running_partial",
                    symbol=symbol,
                )
                last_checkpoint = monotonic()
                if stop_requested.requested:
                    interrupted = True
                    break

            if interrupted:
                summary = _save_checkpoint(
                    symbol_dir=symbol_dir,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    history=history,
                    best_f1=best_f1,
                    best_state=best_state,
                    next_epoch=min(len(history), config.pretrain_epochs),
                    status="interrupted_partial",
                    symbol=symbol,
                )
                results[symbol] = summary
                break

            final_state = best_state or _cpu_state_dict(backbone)
            torch.save(final_state, final_path)
            model.backbone.load_state_dict(final_state, strict=False)
            train_eval_loader = DataLoader(train_ds, shuffle=False, **eval_loader_kwargs)
            split_metrics = {
                "train": _evaluate_classifier(model, train_eval_loader, config.device),
                "val": _evaluate_classifier(model, val_loader, config.device),
                "test": _evaluate_classifier(model, test_loader, config.device),
            }
            summary = _save_checkpoint(
                symbol_dir=symbol_dir,
                config=config,
                model=model,
                optimizer=optimizer,
                history=history,
                best_f1=best_f1,
                best_state=best_state or final_state,
                next_epoch=config.pretrain_epochs,
                status="completed",
                symbol=symbol,
            )
            summary.update(
                {
                    "class_weights": [float(value) for value in class_weights_cpu.tolist()],
                    "train_dataset_class_counts": train_ds.class_counts(),
                    "val_dataset_class_counts": val_ds.class_counts(),
                    "test_dataset_class_counts": test_ds.class_counts(),
                    "split_metrics": split_metrics,
                }
            )
            save_json(summary_path, summary)
            results[symbol] = summary
    finally:
        signal.signal(signal.SIGUSR1, old_usr1)
        signal.signal(signal.SIGTERM, old_term)
    save_json(out_dir / "pretrain_results.json", results)
    return results


@pyrallis.wrap()
def main(config: PretrainConfig) -> None:
    run_pretrain(config)


if __name__ == "__main__":
    main()
