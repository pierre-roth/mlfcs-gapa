from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyrallis
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import PretrainConfig
from .data import PretrainDataset
from .models import PretrainClassifier, build_backbone
from .pipeline import load_symbol_splits, prepare_run
from .utils import ensure_dir, save_json


def run_pretrain(config: PretrainConfig) -> dict[str, dict[str, float | str]]:
    config.apply_mode_defaults()
    out_dir = prepare_run(config)
    results: dict[str, dict[str, float | str]] = {}
    for symbol in config.symbols:
        splits = load_symbol_splits(config, symbol)
        train_ds = PretrainDataset(
            splits["train"],
            lookback=config.lookback,
            horizon=config.pretrain_horizon,
            alpha=config.pretrain_alpha,
            max_samples_per_day=config.max_pretrain_samples_per_day,
        )
        val_ds = PretrainDataset(
            splits["val"],
            lookback=config.lookback,
            horizon=config.pretrain_horizon,
            alpha=config.pretrain_alpha,
            max_samples_per_day=max(512, (config.max_pretrain_samples_per_day or 512) // 4),
        )
        train_loader = DataLoader(train_ds, batch_size=config.pretrain_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.pretrain_batch_size)
        backbone = build_backbone(config.pretrain_backbone, config.lookback).to(config.device)
        model = PretrainClassifier(backbone).to(config.device)
        optimizer = Adam(model.parameters(), lr=config.pretrain_lr)
        criterion = nn.CrossEntropyLoss()
        history = []
        best_f1 = -1.0
        best_state = None
        for epoch in range(config.pretrain_epochs):
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
            model.eval()
            preds = []
            targets = []
            with torch.no_grad():
                for lob, label in val_loader:
                    logits = model(lob.to(config.device))
                    preds.extend(logits.argmax(dim=-1).cpu().tolist())
                    targets.extend(label.tolist())
            precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss / max(len(train_ds), 1),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
            history.append(epoch_record)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_state = {key: value.cpu() for key, value in backbone.state_dict().items()}
        assert best_state is not None
        symbol_dir = ensure_dir(out_dir / symbol / "pretrain")
        torch.save(best_state, symbol_dir / config.save_backbone_name)
        pd.DataFrame(history).to_csv(symbol_dir / "history.csv", index=False)
        summary = {
            "symbol": symbol,
            "backbone_name": config.save_backbone_name,
            "backbone": config.pretrain_backbone,
            "best_f1": best_f1,
            "path": str(symbol_dir / config.save_backbone_name),
        }
        save_json(symbol_dir / "summary.json", summary)
        results[symbol] = summary
    save_json(out_dir / "pretrain_results.json", results)
    return results


@pyrallis.wrap()
def main(config: PretrainConfig) -> None:
    run_pretrain(config)


if __name__ == "__main__":
    main()
