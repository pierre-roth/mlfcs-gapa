from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyrallis
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from .config import PretrainConfig
from .data import DayData, load_splits
from .models import PretrainClassifier, build_backbone
from .utils import ensure_dir, save_json


def _samples_from_days(days: list[DayData], config: PretrainConfig) -> TensorDataset:
    lob_rows = []
    labels = []
    for day in days:
        for idx in day.valid_label_indices(config.lookback, config.pretrain_horizon):
            start = idx - config.lookback + 1
            lob_rows.append(day.normalized_lob[start : idx + 1])
            labels.append(int(day.labels[idx]))
    return TensorDataset(torch.tensor(np.stack(lob_rows), dtype=torch.float32), torch.tensor(labels, dtype=torch.long))


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for lob, y in loader:
            lob = lob.to(device)
            out = model(lob).cpu()
            logits.append(out)
            labels.append(y)
    pred = torch.cat(logits).argmax(dim=-1).numpy()
    true = torch.cat(labels).numpy()
    return {"f1": float(f1_score(true, pred, average="macro"))}


def run_pretrain(config: PretrainConfig) -> dict[str, dict[str, float]]:
    config.apply_mode_defaults()
    output_root = ensure_dir(config.output_dir())
    summaries: dict[str, dict[str, float]] = {}
    for symbol in config.symbols:
        splits = load_splits(config, symbol)
        train_ds = _samples_from_days(splits["train"], config)
        val_days = splits["val"] or splits["test"]
        val_ds = _samples_from_days(val_days, config)
        test_ds = _samples_from_days(splits["test"], config)
        train_loader = DataLoader(train_ds, batch_size=config.pretrain_batch_size, shuffle=True, num_workers=config.pretrain_num_workers)
        val_loader = DataLoader(val_ds, batch_size=config.pretrain_batch_size)
        test_loader = DataLoader(test_ds, batch_size=config.pretrain_batch_size)
        backbone = build_backbone(config.pretrain_backbone, config.lookback)
        model = PretrainClassifier(backbone).to(config.device)
        opt = Adam(model.parameters(), lr=config.pretrain_lr)
        criterion = nn.CrossEntropyLoss()
        symbol_dir = ensure_dir(output_root / symbol / "pretrain")
        best_f1 = float("-inf")
        history = []
        best_state = None
        for epoch in range(config.pretrain_epochs):
            model.train()
            losses = []
            for lob, y in train_loader:
                lob = lob.to(config.device)
                y = y.to(config.device)
                logits = model(lob)
                loss = criterion(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            val_metrics = _evaluate(model, val_loader, config.device)
            history.append({"epoch": epoch, "loss": float(np.mean(losses) if losses else 0.0), **val_metrics})
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_state = {key: value.detach().cpu() for key, value in backbone.state_dict().items()}
        if best_state is None:
            raise RuntimeError("Pretraining produced no checkpoint")
        torch.save(best_state, symbol_dir / config.backbone_name)
        pd.DataFrame(history).to_csv(symbol_dir / "history.csv", index=False)
        backbone.load_state_dict(best_state)
        final_model = PretrainClassifier(backbone).to(config.device)
        final_model.backbone.load_state_dict(best_state)
        test_metrics = _evaluate(final_model, test_loader, config.device)
        summary = {"best_f1": float(best_f1), "test_f1": float(test_metrics["f1"])}
        save_json(symbol_dir / "summary.json", summary)
        summaries[symbol] = summary
    save_json(output_root / "pretrain_summary.json", summaries)
    return summaries


@pyrallis.wrap()
def main(config: PretrainConfig) -> None:
    run_pretrain(config)


if __name__ == "__main__":
    main()

