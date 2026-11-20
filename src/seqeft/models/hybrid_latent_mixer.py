# SPDX-License-Identifier: GPL-3.0-only
# 从论文研究实现移入；仅调整包内导入和公开说明，保留计算与参数名称。
"""Hybrid V1 的 local convolution + global attention temporal mixer。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HybridLatentMixerConfig:
    """latent temporal mixer 的固定容量。"""

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
                raise ValueError("{} 必须是正整数".format(name))
        if self.sequence_length != 256:
            raise ValueError("Hybrid latent mixer 的 sequence_length 必须是 256")
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim 必须能被 heads 整除")
        if self.local_expansion_dim != 2 * self.hidden_dim:
            raise ValueError("local_expansion_dim 必须等于 2D")
        if self.ffn_dim != 4 * self.hidden_dim:
            raise ValueError("ffn_dim 必须等于 4D")
        if self.kernel_size <= 0 or self.kernel_size % 2 != 1:
            raise ValueError("kernel_size 必须是正奇数")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")


class HybridLatentMixerBlock(nn.Module):
    """一个 local Conv、global self-attention、FFN 三段式残差 block。"""

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
    """在 ``[B,256,D]`` 上执行 Hybrid temporal mixing。"""

    def __init__(
        self,
        config: HybridLatentMixerConfig | None = None,
        **config_kwargs,
    ) -> None:
        super().__init__()
        if config is not None and config_kwargs:
            raise TypeError("传入 config 后不能再传 mixer 配置关键字")
        self.config = config or HybridLatentMixerConfig(**config_kwargs)
        self.blocks = nn.ModuleList(
            HybridLatentMixerBlock(self.config) for _ in range(self.config.depth)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        expected = (self.config.sequence_length, self.config.hidden_dim)
        if not torch.is_tensor(value) or value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "Hybrid mixer 输入必须是 [B,{},{}]，实际为 {}".format(
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
]
