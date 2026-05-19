"""LOB pretraining model baselines for Table I."""

from __future__ import annotations

import torch
from torch import nn

from mlfcs_gapa.paper.constants import PAPER


class FCLOBClassifier(nn.Module):
    """Fully connected LOB baseline described as FC-LOB in the paper."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(PAPER.window_length * PAPER.lob_width, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 3),
        )

    def forward(self, lob_window: torch.Tensor) -> torch.Tensor:
        return self.net(lob_window)


class ConvLOBClassifier(nn.Module):
    """Dilated convolutional LOB baseline.

    The paper only says Conv-LOB is a fully convolutional network using dilated
    convolution similar to WaveNet. This implementation keeps that behavior and
    uses global average pooling for the 3-class head.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(PAPER.lob_width, 64, kernel_size=3, padding=1, dilation=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.LeakyReLU(0.01),
            nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=8),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 3),
        )

    def forward(self, lob_window: torch.Tensor) -> torch.Tensor:
        return self.net(lob_window.transpose(1, 2))


class DeepLOBClassifier(nn.Module):
    """CNN-LSTM DeepLOB-style baseline.

    DeepLOB in the cited paper uses a convolutional front-end over 10 LOB levels
    followed by recurrent temporal aggregation. This implementation mirrors the
    convolutional reduction used by Attn-LOB and replaces attention with an LSTM.
    """

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(1, 5), stride=(1, 5)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(1, 4)),
            nn.LeakyReLU(0.01),
        )
        self.inception = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
                    nn.LeakyReLU(0.01),
                ),
                nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
                    nn.LeakyReLU(0.01),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                    nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
                    nn.LeakyReLU(0.01),
                ),
            ]
        )
        self.lstm = nn.LSTM(input_size=192, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, lob_window: torch.Tensor) -> torch.Tensor:
        x = lob_window.unsqueeze(1)
        x = self.conv_stack(x)
        x = torch.cat([branch(x) for branch in self.inception], dim=1)
        sequence = x.squeeze(-1).transpose(1, 2)
        output, _ = self.lstm(sequence)
        return self.classifier(output[:, -1, :])


def make_pretrain_model(name: str) -> nn.Module:
    normalized = name.lower().replace("_", "-")
    if normalized == "fc-lob":
        return FCLOBClassifier()
    if normalized == "conv-lob":
        return ConvLOBClassifier()
    if normalized == "deeplob":
        return DeepLOBClassifier()
    if normalized == "attn-lob":
        from mlfcs_gapa.models.attn_lob import AttnLOBClassifier

        return AttnLOBClassifier()
    raise ValueError(f"unknown pretraining model: {name}")
