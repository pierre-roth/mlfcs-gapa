"""PPO training utilities for the paper's continuous-action agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from mlfcs_gapa.models.attn_lob import AttnLOBEncoder
from mlfcs_gapa.paper.constants import PAPER


class AttnLOBFeatureExtractor(BaseFeaturesExtractor):
    """Stable-Baselines3 feature extractor for the paper C-PPO observation."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        encoder_checkpoint: str | None = None,
        freeze_encoder: bool = False,
        lob_mode: str = "attn",
        use_dynamic_state: bool = True,
        use_agent_state: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.lob_mode = _validate_lob_mode(lob_mode)
        self.use_dynamic_state = use_dynamic_state
        self.use_agent_state = use_agent_state

        lob_dim = 0
        if self.lob_mode == "attn":
            self.lob_encoder: nn.Module = AttnLOBEncoder()
            if encoder_checkpoint:
                load_attn_lob_encoder(self.lob_encoder, Path(encoder_checkpoint))
            if freeze_encoder:
                for parameter in self.lob_encoder.parameters():
                    parameter.requires_grad = False
            lob_dim = 64
        elif self.lob_mode == "mlp":
            if encoder_checkpoint:
                raise ValueError("encoder_checkpoint is only valid with lob_mode='attn'")
            self.lob_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(PAPER.window_length * PAPER.lob_width, 64),
                nn.LeakyReLU(0.01),
            )
            lob_dim = 64
        else:
            if encoder_checkpoint:
                raise ValueError("encoder_checkpoint is only valid with lob_mode='attn'")
            self.lob_encoder = nn.Identity()

        dynamic_dim = int(observation_space["dynamic_state"].shape[0]) if use_dynamic_state else 0
        agent_dim = int(observation_space["agent_state"].shape[0]) if use_agent_state else 0
        if lob_dim + dynamic_dim + agent_dim == 0:
            raise ValueError("at least one observation component must be enabled")
        self.projection = nn.Sequential(
            nn.Linear(lob_dim + dynamic_dim + agent_dim, features_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        if self.lob_mode != "none":
            features.append(self.lob_encoder(observations["lob_state"].float()))
        if self.use_dynamic_state:
            features.append(observations["dynamic_state"].float())
        if self.use_agent_state:
            features.append(observations["agent_state"].float())
        return self.projection(torch.cat(features, dim=1))


def load_attn_lob_encoder(encoder: AttnLOBEncoder, checkpoint_path: Path) -> None:
    """Load encoder weights from a raw encoder or AttnLOBClassifier checkpoint."""

    checkpoint: Any = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise ValueError(f"unsupported checkpoint format: {checkpoint_path}")

    encoder_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state[key.removeprefix("encoder.")] = value
        elif key.startswith("lob_encoder."):
            encoder_state[key.removeprefix("lob_encoder.")] = value
        elif key.startswith("classifier."):
            continue
        else:
            encoder_state[key] = value

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected encoder checkpoint keys: {unexpected}")
    if missing:
        raise ValueError(f"missing encoder checkpoint keys: {sorted(missing)}")


def _validate_lob_mode(lob_mode: str) -> str:
    if lob_mode not in {"attn", "mlp", "none"}:
        raise ValueError("lob_mode must be one of: attn, mlp, none")
    return lob_mode
