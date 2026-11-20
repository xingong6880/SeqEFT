# SPDX-License-Identifier: GPL-3.0-only
"""Temporal mixers built from convolution, attention, and feed-forward layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HybridLatentMixerConfig:

    hidden_dim: int = 256
    depth: int = 4
    heads: int = 8
    ffn_dim: int = 1024
    local_expansion_dim: int = 512
    kernel_size: int = 5
    dropout: float = 0.1
    sequence_length: int = 256

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim",
            "depth",
            "heads",
            "ffn_dim",
            "local_expansion_dim",
            "sequence_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if self.sequence_length != 256:
            raise ValueError("sequence_length must be 256")
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.local_expansion_dim != 2 * self.hidden_dim:
            raise ValueError("local_expansion_dim must be twice hidden_dim")
        if self.ffn_dim != 4 * self.hidden_dim:
            raise ValueError("ffn_dim must be four times hidden_dim")
        if self.kernel_size <= 0 or self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class HybridLatentMixerBlock(nn.Module):
    """Residual block with local convolution, global self-attention, and an FFN."""

    def __init__(self, config: HybridLatentMixerConfig) -> None:
        super().__init__()
        width = config.hidden_dim
        self.local_norm = nn.LayerNorm(width)
        self.depthwise_conv = nn.Conv1d(
            width,
            width,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
            groups=width,
        )
        self.local_expand = nn.Conv1d(width, config.local_expansion_dim, kernel_size=1)
        self.local_activation = nn.GELU()
        self.local_project = nn.Conv1d(config.local_expansion_dim, width, kernel_size=1)
        self.local_dropout = nn.Dropout(config.dropout)

        self.global_norm = nn.LayerNorm(width)
        self.global_attention = nn.MultiheadAttention(
            width,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.global_dropout = nn.Dropout(config.dropout)

        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, width),
            nn.Dropout(config.dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        local = self.local_norm(value).transpose(1, 2)
        local = self.depthwise_conv(local)
        local = self.local_project(self.local_activation(self.local_expand(local)))
        value = value + self.local_dropout(local.transpose(1, 2))

        normalized = self.global_norm(value)
        attended = self.global_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        value = value + self.global_dropout(attended)
        return value + self.ffn(self.ffn_norm(value))


class HybridLatentTemporalMixer(nn.Module):

    def __init__(
        self,
        config: HybridLatentMixerConfig | None = None,
        **config_kwargs,
    ) -> None:
        super().__init__()
        if config is not None and config_kwargs:
            raise TypeError("Pass either config or mixer keyword settings")
        self.config = config or HybridLatentMixerConfig(**config_kwargs)
        self.blocks = nn.ModuleList(
            HybridLatentMixerBlock(self.config) for _ in range(self.config.depth)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        expected = (self.config.sequence_length, self.config.hidden_dim)
        if not torch.is_tensor(value) or value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "Expected mixer input shape [B, {}, {}], got {}".format(
                    *expected,
                    tuple(value.shape) if torch.is_tensor(value) else type(value),
                )
            )
        for block in self.blocks:
            value = block(value)
        return value


class LocalOnlyLatentMixerBlock(nn.Module):
    """Reuse the initialized local and feed-forward layers."""

    def __init__(self, source: HybridLatentMixerBlock) -> None:
        super().__init__()
        if not isinstance(source, HybridLatentMixerBlock):
            raise TypeError("source must be a HybridLatentMixerBlock")
        # Transfer the existing modules to preserve parameter names and initialization.
        self.local_norm = source.local_norm
        self.depthwise_conv = source.depthwise_conv
        self.local_expand = source.local_expand
        self.local_activation = source.local_activation
        self.local_project = source.local_project
        self.local_dropout = source.local_dropout
        self.ffn_norm = source.ffn_norm
        self.ffn = source.ffn

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        local = self.local_norm(value).transpose(1, 2)
        local = self.depthwise_conv(local)
        local = self.local_project(self.local_activation(self.local_expand(local)))
        value = value + self.local_dropout(local.transpose(1, 2))
        return value + self.ffn(self.ffn_norm(value))


class LocalOnlyLatentTemporalMixer(nn.Module):
    """Mix the reference latent sequence with local blocks."""

    def __init__(
        self,
        config: HybridLatentMixerConfig,
        blocks: list[LocalOnlyLatentMixerBlock],
    ) -> None:
        super().__init__()
        if len(blocks) != config.depth:
            raise ValueError("The number of blocks must match config.depth")
        self.config = config
        self.blocks = nn.ModuleList(blocks)

    @classmethod
    def from_hybrid(
        cls, mixer: HybridLatentTemporalMixer
    ) -> tuple["LocalOnlyLatentTemporalMixer", list[dict[str, int | str]]]:
        """Keep the local blocks and report the removed global parameters."""
        if not isinstance(mixer, HybridLatentTemporalMixer):
            raise TypeError("mixer must be a HybridLatentTemporalMixer")

        removed = []
        for name, parameter in mixer.named_parameters():
            if ".global_norm." in name or ".global_attention." in name:
                removed.append({"name": name, "numel": int(parameter.numel())})
        blocks = [LocalOnlyLatentMixerBlock(block) for block in mixer.blocks]
        result = cls(mixer.config, blocks)
        return result, removed

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        expected = (self.config.sequence_length, self.config.hidden_dim)
        if not torch.is_tensor(value) or value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "Expected local mixer input shape [B, {}, {}], got {}".format(
                    *expected,
                    tuple(value.shape) if torch.is_tensor(value) else type(value),
                )
            )
        for block in self.blocks:
            value = block(value)
        return value


HybridLatentMixer = HybridLatentTemporalMixer

__all__ = [
    "HybridLatentMixer",
    "HybridLatentMixerBlock",
    "HybridLatentMixerConfig",
    "HybridLatentTemporalMixer",
    "LocalOnlyLatentMixerBlock",
    "LocalOnlyLatentTemporalMixer",
]
