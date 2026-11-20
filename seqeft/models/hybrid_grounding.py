# SPDX-License-Identifier: GPL-3.0-only
"""Overlapping local attention from BOLD latent positions to EEG frames."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HybridGroundingConfig:

    hidden_dim: int = 256
    heads: int = 8
    slots_per_frame: int = 16
    input_length: int = 512
    output_length: int = 256
    context_radius: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim",
            "heads",
            "slots_per_frame",
            "input_length",
            "output_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if isinstance(self.context_radius, bool) or not isinstance(self.context_radius, int):
            raise ValueError("context_radius must be a nonnegative integer")
        if self.context_radius < 0:
            raise ValueError("context_radius must be a nonnegative integer")
        if self.input_length != 512 or self.output_length != 256:
            raise ValueError("Grounding requires 512 EEG frames and 256 latent steps")
        if self.input_length != self.output_length * 2:
            raise ValueError("input_length must be twice output_length")
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class OverlappingLocalCrossAttention(nn.Module):
    """Attend over a fixed window around each latent position and mask out-of-range frames."""

    def __init__(
        self,
        config: HybridGroundingConfig | None = None,
        **config_kwargs,
    ) -> None:
        super().__init__()
        if config is not None and config_kwargs:
            raise TypeError("Pass either config or grounding keyword settings")
        self.config = config or HybridGroundingConfig(**config_kwargs)
        width = self.config.hidden_dim
        self.position_queries = nn.Parameter(
            torch.empty(1, self.config.output_length, width)
        )
        nn.init.trunc_normal_(self.position_queries, std=0.02)
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width,
            self.config.heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(width)
        offsets = torch.arange(
            -self.config.context_radius,
            self.config.context_radius + 2,
            dtype=torch.long,
        )
        self.register_buffer("window_offsets", offsets, persistent=False)

    def window_indices_and_mask(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return window indices and the mask for out-of-range EEG slots."""
        centers = torch.arange(
            self.config.output_length,
            device=device,
            dtype=torch.long,
        ).mul(2).unsqueeze(1)
        raw_indices = centers + self.window_offsets.to(device).unsqueeze(0)
        valid = (raw_indices >= 0) & (raw_indices < self.config.input_length)
        # Clamp for safe indexing. The mask hides all out-of-range entries.
        indices = raw_indices.clamp(0, self.config.input_length - 1)
        invalid = (~valid).unsqueeze(-1).expand(
            -1,
            -1,
            self.config.slots_per_frame,
        )
        return indices, invalid.reshape(self.config.output_length, -1)

    def make_query(self, value: torch.Tensor) -> torch.Tensor:
        return self.query_norm(self.position_queries).expand(value.shape[0], -1, -1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Map [B, 512, slots, D] EEG tokens to [B, 256, D] latents."""
        expected = (
            self.config.input_length,
            self.config.slots_per_frame,
            self.config.hidden_dim,
        )
        if not torch.is_tensor(value) or value.ndim != 4 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "Expected grounding input shape [B, 512, {}, {}], got {}".format(
                    self.config.slots_per_frame,
                    self.config.hidden_dim,
                    tuple(value.shape) if torch.is_tensor(value) else type(value),
                )
            )

        batch = value.shape[0]
        indices, invalid = self.window_indices_and_mask(value.device)
        # [B,256,W,K,D] -> [B*256,W*K,D]
        local = value[:, indices]
        local = local.reshape(
            batch * self.config.output_length,
            -1,
            self.config.hidden_dim,
        )
        key = self.key_norm(local)
        query = self.make_query(value).reshape(
            batch * self.config.output_length, 1, self.config.hidden_dim
        )
        key_padding_mask = invalid.unsqueeze(0).expand(batch, -1, -1).reshape(
            batch * self.config.output_length,
            -1,
        )
        attended = self.cross_attention(
            query,
            key,
            key,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        output = self.output_norm(attended + query)
        return output[:, 0, :].reshape(
            batch,
            self.config.output_length,
            self.config.hidden_dim,
        )


class ContentQueryGrounding(OverlappingLocalCrossAttention):
    """Combine position queries with projected EEG features from adjacent frame pairs."""

    def __init__(self, config: HybridGroundingConfig | None = None, **config_kwargs) -> None:
        super().__init__(config, **config_kwargs)
        self.content_projection = nn.Linear(
            2 * self.config.hidden_dim,
            self.config.hidden_dim,
            bias=False,
        )
        nn.init.zeros_(self.content_projection.weight)

    @classmethod
    def from_position_grounding(
        cls, grounding: OverlappingLocalCrossAttention
    ) -> "ContentQueryGrounding":
        """Copy the base attention weights while preserving the caller's RNG state."""
        if not isinstance(grounding, OverlappingLocalCrossAttention):
            raise TypeError("grounding must be an OverlappingLocalCrossAttention module")
        with torch.random.fork_rng(devices=[]):
            result = cls(grounding.config)
        loaded = result.load_state_dict(grounding.state_dict(), strict=False)
        if loaded.missing_keys != ["content_projection.weight"] or loaded.unexpected_keys:
            raise RuntimeError("Content grounding and position grounding have different shared parameters")
        return result

    @staticmethod
    def pair_frame_content(value: torch.Tensor) -> torch.Tensor:
        """Concatenate the mean token features of each ordered pair of EEG frames."""
        frame_content = value.mean(dim=2)
        return torch.cat(
            (frame_content[:, 0::2], frame_content[:, 1::2]), dim=-1
        )

    def make_query(self, value: torch.Tensor) -> torch.Tensor:
        """Add the frame-pair projection to each latent position query."""
        paired_content = self.pair_frame_content(value)
        content = self.content_projection(paired_content)
        return self.query_norm(self.position_queries + content).expand(
            value.shape[0], -1, -1
        )


HybridLocalGrounding = OverlappingLocalCrossAttention

__all__ = [
    "ContentQueryGrounding",
    "HybridGroundingConfig",
    "HybridLocalGrounding",
    "OverlappingLocalCrossAttention",
]
