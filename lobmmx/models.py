from __future__ import annotations

import copy

import torch
from torch import nn
from torch.distributions import Beta


class AttnLOB(nn.Module):
    def __init__(self, lookback: int = 50, output_dim: int = 64, num_heads: int = 10, key_dim: int = 16) -> None:
        super().__init__()
        self.lookback = lookback
        self.output_dim = output_dim
        self.attn_dim = num_heads * key_dim
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 32, (1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (1, 5), stride=(1, 5)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (1, 4)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding="same"),
            nn.LeakyReLU(0.01),
        )
        self.branch_3 = nn.Sequential(nn.Conv2d(32, 64, (1, 1)), nn.LeakyReLU(0.01), nn.Conv2d(64, 64, (3, 1), padding=(1, 0)), nn.LeakyReLU(0.01))
        self.branch_5 = nn.Sequential(nn.Conv2d(32, 64, (1, 1)), nn.LeakyReLU(0.01), nn.Conv2d(64, 64, (5, 1), padding=(2, 0)), nn.LeakyReLU(0.01))
        self.branch_pool = nn.Sequential(nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)), nn.Conv2d(32, 64, (1, 1)), nn.LeakyReLU(0.01))
        self.temporal_proj = nn.Linear(192, self.attn_dim)
        self.attn = nn.MultiheadAttention(self.attn_dim, num_heads=num_heads, batch_first=True)
        self.output_proj = nn.Linear(self.attn_dim, output_dim)
        self.last_attention: torch.Tensor | None = None

    def features(self, lob: torch.Tensor) -> torch.Tensor:
        x = lob.unsqueeze(1)
        x = self.spatial(x)
        x = torch.cat([self.branch_3(x), self.branch_5(x), self.branch_pool(x)], dim=1)
        x = x.squeeze(-1).transpose(1, 2)
        x = self.temporal_proj(x)
        query = x[:, -1:, :]
        attn_out, attn_weights = self.attn(query, x, x, need_weights=True, average_attn_weights=False)
        self.last_attention = attn_weights.detach()
        return self.output_proj(attn_out.squeeze(1))

    def forward(self, lob: torch.Tensor) -> torch.Tensor:
        return self.features(lob)


class SimpleLOB(nn.Module):
    def __init__(self, lookback: int = 50, input_dim: int = 40, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Linear(16, output_dim), nn.LeakyReLU(0.01))

    def features(self, lob: torch.Tensor) -> torch.Tensor:
        x = lob.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)

    def forward(self, lob: torch.Tensor) -> torch.Tensor:
        return self.features(lob)


class PretrainClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.output_dim, num_classes)

    def forward(self, lob: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.features(lob))


class PretrainMultiTask(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = backbone
        self.mid_head = nn.Linear(backbone.output_dim, num_classes)
        self.spread_head = nn.Linear(backbone.output_dim, num_classes)
        self.flow_head = nn.Linear(backbone.output_dim, num_classes)

    def forward(self, lob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone.features(lob)
        return self.mid_head(features), self.spread_head(features), self.flow_head(features)


class SharedStateEncoder(nn.Module):
    def __init__(self, backbone: nn.Module | None, flat_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        backbone_dim = 0 if backbone is None else backbone.output_dim
        self.trunk = nn.Sequential(
            nn.Linear(backbone_dim + flat_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
        )
        self.output_dim = hidden_dim

    def forward(self, lob: torch.Tensor | None, flat: torch.Tensor) -> torch.Tensor:
        parts = [flat]
        if self.backbone is not None and lob is not None:
            parts.insert(0, self.backbone.features(lob))
        return self.trunk(torch.cat(parts, dim=-1))


class ContinuousActorCritic(nn.Module):
    def __init__(self, encoder: SharedStateEncoder, action_dim: int = 3) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        self.alpha_head = nn.Linear(encoder.output_dim, action_dim)
        self.beta_head = nn.Linear(encoder.output_dim, action_dim)
        self.value_head = nn.Linear(encoder.output_dim, 1)

    def forward(self, lob: torch.Tensor | None, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(lob, flat)
        alpha = torch.nn.functional.softplus(self.alpha_head(z)) + 1.0
        beta = torch.nn.functional.softplus(self.beta_head(z)) + 1.0
        value = self.value_head(z).squeeze(-1)
        return alpha, beta, value

    def dist_value(self, lob: torch.Tensor | None, flat: torch.Tensor) -> tuple[Beta, torch.Tensor]:
        alpha, beta, value = self.forward(lob, flat)
        return Beta(alpha, beta), value


class DuelingQNetwork(nn.Module):
    def __init__(self, encoder: SharedStateEncoder, num_actions: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.value = nn.Linear(encoder.output_dim, 1)
        self.advantage = nn.Linear(encoder.output_dim, num_actions)

    def forward(self, lob: torch.Tensor | None, flat: torch.Tensor) -> torch.Tensor:
        z = self.encoder(lob, flat)
        value = self.value(z)
        advantage = self.advantage(z)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


def build_backbone(name: str, lookback: int) -> nn.Module:
    if name == "simple":
        return SimpleLOB(lookback=lookback)
    return AttnLOB(lookback=lookback)


def clone_module(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)
