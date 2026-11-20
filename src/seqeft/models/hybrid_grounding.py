# SPDX-License-Identifier: GPL-3.0-only
# 从论文研究实现移入；仅调整包内导入和公开说明，保留计算与参数名称。
"""Hybrid V1 的 overlapping local EEG-to-latent grounding。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HybridGroundingConfig:
    """固定的 local grounding shape contract。"""

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
                raise ValueError("{} 必须是正整数".format(name))
        if isinstance(self.context_radius, bool) or not isinstance(self.context_radius, int):
            raise ValueError("context_radius 必须是非负整数")
        if self.context_radius < 0:
            raise ValueError("context_radius 必须是非负整数")
        if self.input_length != 512 or self.output_length != 256:
            raise ValueError("Hybrid grounding 固定为 512 -> 256")
        if self.input_length != self.output_length * 2:
            raise ValueError("input_length 必须是 output_length 的两倍")
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim 必须能被 heads 整除")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")


class OverlappingLocalCrossAttention(nn.Module):
    """让每个 BOLD latent position 读取 overlapping EEG local window。

    窗口使用固定长度，越过序列边界的 token 由 ``key_padding_mask`` 屏蔽。
    这样既保持 batch shape，又不会通过 clipping 重复边界 frame 或产生 wrap-around。
    """

    def __init__(
        self,
        config: HybridGroundingConfig | None = None,
        **config_kwargs,
    ) -> None:
        super().__init__()
        if config is not None and config_kwargs:
            raise TypeError("传入 config 后不能再传 grounding 配置关键字")
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
        """返回 ``[256,W]`` frame indices 与 ``[256,W*K]`` invalid mask。"""
        centers = torch.arange(
            self.config.output_length,
            device=device,
            dtype=torch.long,
        ).mul(2).unsqueeze(1)
        raw_indices = centers + self.window_offsets.to(device).unsqueeze(0)
        valid = (raw_indices >= 0) & (raw_indices < self.config.input_length)
        # 仅为满足 gather 的固定 shape 做 clipping；所有无效位置都会被 mask。
        indices = raw_indices.clamp(0, self.config.input_length - 1)
        invalid = (~valid).unsqueeze(-1).expand(
            -1,
            -1,
            self.config.slots_per_frame,
        )
        return indices, invalid.reshape(self.config.output_length, -1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """将 ``[B,512,16,D]`` 映射为 ``[B,256,D]``。"""
        expected = (
            self.config.input_length,
            self.config.slots_per_frame,
            self.config.hidden_dim,
        )
        if not torch.is_tensor(value) or value.ndim != 4 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "local grounding 输入必须是 [B,512,{},{}]，实际为 {}".format(
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
        query = self.query_norm(self.position_queries).expand(batch, -1, -1)
        query = query.reshape(batch * self.config.output_length, 1, self.config.hidden_dim)
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


HybridLocalGrounding = OverlappingLocalCrossAttention

__all__ = [
    "HybridGroundingConfig",
    "HybridLocalGrounding",
    "OverlappingLocalCrossAttention",
]
