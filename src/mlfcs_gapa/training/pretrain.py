"""Supervised mid-price direction pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlfcs_gapa.data.pretraining import PretrainArrays


@dataclass(frozen=True)
class PretrainMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    train_loss: float
    n_train: int
    n_val: int


def train_lob_classifier(
    model: nn.Module,
    arrays: PretrainArrays,
    *,
    evaluation_arrays: PretrainArrays | None = None,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 1,
    device: str = "cpu",
) -> PretrainMetrics:
    """Train a 3-class LOB classifier and return Table I-style metrics."""

    if len(arrays.x) != len(arrays.y):
        raise ValueError("x and y must have the same length")
    if len(arrays.x) < 10:
        raise ValueError("not enough samples for pretraining")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(arrays.x))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(indices) * validation_fraction)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if len(train_idx) == 0:
        raise ValueError("validation split leaves no training samples")

    x_train = torch.from_numpy(arrays.x[train_idx]).float()
    y_train = torch.from_numpy(arrays.y[train_idx]).long()
    if evaluation_arrays is None:
        x_eval = torch.from_numpy(arrays.x[val_idx]).float()
        y_eval = torch.from_numpy(arrays.y[val_idx]).long()
    else:
        if len(evaluation_arrays.x) != len(evaluation_arrays.y):
            raise ValueError("evaluation x and y must have the same length")
        if len(evaluation_arrays.x) == 0:
            raise ValueError("evaluation arrays must not be empty")
        x_eval = torch.from_numpy(evaluation_arrays.x).float()
        y_eval = torch.from_numpy(evaluation_arrays.y).long()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else 0.0

    model.eval()
    prediction_batches: list[np.ndarray] = []
    eval_loader = DataLoader(TensorDataset(x_eval), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in eval_loader:
            logits = model(batch_x.to(device))
            prediction_batches.append(logits.argmax(dim=1).cpu().numpy())
    predictions = np.concatenate(prediction_batches)

    labels = y_eval.numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    accuracy = float((predictions == labels).mean())

    return PretrainMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=accuracy,
        train_loss=last_loss,
        n_train=int(len(train_idx)),
        n_val=int(len(labels)),
    )
