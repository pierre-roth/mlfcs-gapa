from __future__ import annotations

import json
from pathlib import Path

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
from .pipeline import load_symbol_splits, prepare_run
from .utils import save_json
from lobmmx.models import PretrainClassifier, build_backbone


def _dataset_class_weights(dataset: PretrainDataset) -> torch.Tensor:
    counts = dataset.class_counts()
    safe = torch.tensor([max(counts[idx], 1) for idx in range(3)], dtype=torch.float32)
    weights = safe.sum() / safe
    return weights / weights.mean()


def _dataset_sampler(dataset: PretrainDataset) -> WeightedRandomSampler | None:
    if len(dataset) == 0:
        return None
    class_weights = _dataset_class_weights(dataset)
    sample_weights = class_weights[torch.from_numpy(dataset.sample_labels_np)]
    return WeightedRandomSampler(sample_weights.double(), num_samples=len(dataset), replacement=True)


def _evaluate_classifier(model: PretrainClassifier, loader: DataLoader, device: str) -> dict[str, float | int | list[int]]:
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for lob, labels in loader:
            logits = model(lob.to(device))
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(labels.tolist())
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
    accuracy = float(sum(int(p == t) for p, t in zip(preds, targets, strict=True)) / len(targets))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": accuracy,
        "target_class_counts": [int(value) for value in np.bincount(np.asarray(targets, dtype=np.int64), minlength=3)],
        "predicted_class_counts": [int(value) for value in np.bincount(np.asarray(preds, dtype=np.int64), minlength=3)],
        "confusion_matrix": confusion_matrix(targets, preds, labels=[0, 1, 2]).astype(int).tolist(),
        "samples": int(len(targets)),
    }


def run_pretrain(config: PretrainConfig) -> dict[str, dict[str, float | str]]:
    config.apply_mode_defaults()
    out_dir = prepare_run(config, label="pretrain")
    summaries: dict[str, dict[str, float | str]] = {}
    for symbol in config.symbols:
        symbol_dir = Path(out_dir) / symbol / "pretrain"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        splits = load_symbol_splits(config, symbol)
        train_ds = PretrainDataset(splits["train"], config.lookback, config.pretrain_horizon, config.pretrain_alpha, config.max_pretrain_samples_per_day)
        val_ds = PretrainDataset(splits["val"] or splits["test"], config.lookback, config.pretrain_horizon, config.pretrain_alpha, config.max_pretrain_samples_per_day)
        test_ds = PretrainDataset(splits["test"], config.lookback, config.pretrain_horizon, config.pretrain_alpha, config.max_pretrain_samples_per_day)
        train_loader = DataLoader(train_ds, batch_size=config.pretrain_batch_size, sampler=_dataset_sampler(train_ds), shuffle=False, num_workers=config.pretrain_num_workers)
        val_loader = DataLoader(val_ds, batch_size=config.pretrain_batch_size, shuffle=False, num_workers=config.pretrain_num_workers)
        test_loader = DataLoader(test_ds, batch_size=config.pretrain_batch_size, shuffle=False, num_workers=config.pretrain_num_workers)
        backbone = build_backbone(config.pretrain_backbone, config.lookback).to(config.device)
        model = PretrainClassifier(backbone).to(config.device)
        optimizer = Adam(model.parameters(), lr=config.pretrain_lr)
        criterion = nn.CrossEntropyLoss(weight=_dataset_class_weights(train_ds).to(config.device))
        best_f1 = -1.0
        best_state = None
        history = []
        for epoch in range(config.pretrain_epochs):
            model.train()
            losses = []
            for lob, labels in train_loader:
                logits = model(lob.to(config.device))
                loss = criterion(logits, labels.to(config.device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            val_metrics = _evaluate_classifier(model, val_loader, config.device)
            epoch_row = {"epoch": epoch, "loss": float(np.mean(losses) if losses else 0.0), **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}}
            history.append(epoch_row)
            if float(val_metrics["f1"]) > best_f1:
                best_f1 = float(val_metrics["f1"])
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        assert best_state is not None
        model.load_state_dict(best_state)
        torch.save(model.backbone.state_dict(), symbol_dir / config.backbone_name)
        pd.DataFrame(history).to_csv(symbol_dir / "history.csv", index=False)
        summary = {
            "symbol": symbol,
            "backbone": config.pretrain_backbone,
            "best_f1": best_f1,
            "path": str(symbol_dir / config.backbone_name),
            "split_metrics": {
                "train": _evaluate_classifier(model, train_loader, config.device),
                "val": _evaluate_classifier(model, val_loader, config.device),
                "test": _evaluate_classifier(model, test_loader, config.device),
            },
        }
        save_json(symbol_dir / "summary.json", summary)
        summaries[symbol] = summary
    save_json(Path(out_dir) / "pretrain_summary.json", summaries)
    return summaries


@pyrallis.wrap()
def main(config: PretrainConfig) -> None:
    run_pretrain(config)


if __name__ == "__main__":
    main()
