# SPDX-License-Identifier: GPL-3.0-only
# 从论文研究实现移入；仅调整包内导入和公开说明，保留计算与参数名称。
"""single-scan5 的局部 EEG 内容 Query；Q0 仍使用原位置 Query 模块。"""

from __future__ import annotations

import torch
from torch import nn

from .hybrid_grounding import HybridGroundingConfig, OverlappingLocalCrossAttention


class ContentQueryGrounding(OverlappingLocalCrossAttention):
    """在原位置 Query 上加一条有序两帧 EEG 内容投影。"""

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
        """复用 Q0 的公共参数，并隔离新增模块初始化的随机数消耗。"""
        if not isinstance(grounding, OverlappingLocalCrossAttention):
            raise TypeError("grounding 必须是原位置 Query 模块")
        with torch.random.fork_rng(devices=[]):
            result = cls(grounding.config)
        loaded = result.load_state_dict(grounding.state_dict(), strict=False)
        if loaded.missing_keys != ["content_projection.weight"] or loaded.unexpected_keys:
            raise RuntimeError("内容 Query 公共参数与位置 Query 参数集合不一致")
        return result

    @staticmethod
    def pair_frame_content(value: torch.Tensor) -> torch.Tensor:
        """按相邻帧顺序拼接各帧的 slot mean。"""
        frame_content = value.mean(dim=2)
        return torch.cat(
            (frame_content[:, 0::2], frame_content[:, 1::2]), dim=-1
        )

    def make_query(self, value: torch.Tensor) -> torch.Tensor:
        """由位置 embedding 和每对有序 EEG 帧构造 ``[B,256,D]`` Query。"""
        paired_content = self.pair_frame_content(value)
        content = self.content_projection(paired_content)
        return self.query_norm(self.position_queries + content).expand(
            value.shape[0], -1, -1
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """保留原 local K/V、边界 mask 与 residual attention 计算。"""
        expected = (
            self.config.input_length,
            self.config.slots_per_frame,
            self.config.hidden_dim,
        )
        if (
            not torch.is_tensor(value)
            or value.ndim != 4
            or tuple(value.shape[1:]) != expected
        ):
            raise ValueError(
                "local grounding 输入必须是 [B,512,{},{}]，实际为 {}".format(
                    self.config.slots_per_frame,
                    self.config.hidden_dim,
                    tuple(value.shape) if torch.is_tensor(value) else type(value),
                )
            )

        batch = value.shape[0]
        indices, invalid = self.window_indices_and_mask(value.device)
        local = value[:, indices].reshape(
            batch * self.config.output_length,
            -1,
            self.config.hidden_dim,
        )
        key = self.key_norm(local)
        query = self.make_query(value).reshape(
            batch * self.config.output_length, 1, self.config.hidden_dim
        )
        key_padding_mask = invalid.unsqueeze(0).expand(batch, -1, -1).reshape(
            batch * self.config.output_length, -1
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


__all__ = ["ContentQueryGrounding"]
