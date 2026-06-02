"""LOB pretraining model baselines for Table I."""

from __future__ import annotations

import torch
from torch import nn

from mlfcs_gapa.paper.constants import PAPER


PAPER_REPORTED_PRETRAIN_PARAMS: dict[str, int] = {
    "FC-LOB": 256_064,
    "Conv-LOB": 172_320,
    "DeepLOB": 139_168,
    "Attn-LOB": 176_320,
}

PAPER_PRETRAIN_INPUT_SHAPES: dict[str, tuple[int, int]] = {
    "FC-LOB": (100, PAPER.lob_width),
    "Conv-LOB": (1024, PAPER.lob_width),
    "DeepLOB": (100, PAPER.lob_width),
    "Attn-LOB": PAPER.lob_window_shape,
}


class FCLOBClassifier(nn.Module):
    """Fully connected LOB baseline matching the released FC-LOB encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.input_shape = PAPER_PRETRAIN_INPUT_SHAPES["FC-LOB"]
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100 * PAPER.lob_width, 64),
            nn.LeakyReLU(0.01),
        )
        self.classifier = nn.Linear(64, 3)

    def forward(self, lob_window: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(lob_window))


class WaveNetResidualBlock(nn.Module):
    """Gated residual block in the style of WaveNet."""

    def __init__(
        self,
        *,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.dilation = dilation
        self.filter_conv = nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1),
        )
        self.gate_conv = nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1),
        )
        self.residual_projection = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_projection = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.shape[-1]
        filtered = self.filter_conv(x)[..., :length]
        gated = self.gate_conv(x)[..., :length]
        z = torch.tanh(filtered) * torch.sigmoid(gated)
        return x + self.residual_projection(z), self.skip_projection(z)


class ConvLOBEncoder(nn.Module):
    """WaveNet-style dilated convolutional encoder for Conv-LOB.

    Public material only states that Conv-LOB is fully convolutional and uses
    WaveNet-like dilations. This encoder keeps the WaveNet essentials: causal
    dilated residual blocks, gated activations, residual paths, skip paths, and
    a convolutional readout. The channel counts are the smallest conventional
    configuration found in the audit that preserves those ingredients while
    matching the paper's encoder parameter count exactly.
    """

    def __init__(
        self,
        *,
        residual_channels: int = 80,
        skip_channels: int = 16,
        layers: int = 5,
        kernel_size: int = 2,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(PAPER.lob_width, residual_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                WaveNetResidualBlock(
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=2**layer_index,
                )
                for layer_index in range(layers)
            ]
        )
        self.post = nn.Sequential(
            nn.LeakyReLU(0.01),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(skip_channels, output_dim, kernel_size=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, lob_window: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(lob_window.transpose(1, 2))
        skip_total = None
        for block in self.blocks:
            x, skip = block(x)
            skip_total = skip if skip_total is None else skip_total + skip
        if skip_total is None:
            raise RuntimeError("ConvLOBEncoder must contain at least one residual block")
        return self.post(skip_total)


class ConvLOBClassifier(nn.Module):
    """Dilated convolutional LOB baseline.

    The paper only says Conv-LOB is a fully convolutional network using dilated
    convolution similar to WaveNet. The chosen gated residual configuration is
    the exact 172,320-parameter encoder reconstruction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_shape = PAPER_PRETRAIN_INPUT_SHAPES["Conv-LOB"]
        self.encoder = ConvLOBEncoder()
        self.classifier = nn.Linear(64, 3)

    def forward(self, lob_window: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(lob_window))


class KerasStyleLSTM(nn.Module):
    """Fused LSTM with Keras-equivalent single-bias parameter counting."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size + 1,
            hidden_size=hidden_size,
            batch_first=True,
            bias=False,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        ones = sequence.new_ones((*sequence.shape[:2], 1))
        output, _ = self.lstm(torch.cat([sequence, ones], dim=-1))
        return output[:, -1, :]


class DeepLOBClassifier(nn.Module):
    """Table-I-compatible CNN-LSTM DeepLOB baseline.

    The original DeepLOB reference implementation supplies the CNN/Inception
    plus LSTM pattern, while the market-making paper's own Attn-LOB code
    supplies a different 40-to-1 LOB width reduction. Combining the paper's
    front-end with DeepLOB's temporal LSTM aggregation is the closest
    count-exact compromise: it keeps the DeepLOB modeling idea and matches the
    Table I encoder parameter count exactly.
    """

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.input_shape = PAPER_PRETRAIN_INPUT_SHAPES["DeepLOB"]
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
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
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
        self.lstm = KerasStyleLSTM(input_size=192, hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, lob_window: torch.Tensor) -> torch.Tensor:
        x = lob_window.unsqueeze(1)
        x = self.conv_stack(x)
        x = torch.cat([branch(x) for branch in self.inception], dim=1)
        sequence = x.squeeze(-1).transpose(1, 2)
        return self.classifier(self.lstm(sequence))


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


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_encoder_parameters(model: nn.Module) -> int:
    encoder = getattr(model, "encoder", None)
    if isinstance(encoder, nn.Module):
        return count_parameters(encoder)
    if isinstance(model, DeepLOBClassifier):
        return (
            count_parameters(model.conv_stack)
            + count_parameters(model.inception)
            + count_parameters(model.lstm)
        )
    return count_parameters(model)


def pretrain_input_shape(name: str) -> tuple[int, int]:
    normalized = name.lower().replace("_", "-")
    for model_name, input_shape in PAPER_PRETRAIN_INPUT_SHAPES.items():
        if model_name.lower().replace("_", "-") == normalized:
            return input_shape
    raise ValueError(f"unknown pretraining model: {name}")


def paper_reported_parameter_count(name: str) -> int | None:
    normalized = name.lower().replace("_", "-")
    for model_name, parameter_count in PAPER_REPORTED_PRETRAIN_PARAMS.items():
        if model_name.lower().replace("_", "-") == normalized:
            return parameter_count
    return None
