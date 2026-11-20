# SPDX-License-Identifier: GPL-3.0-only
# 从论文研究实现移入；仅调整包内导入和公开说明，保留计算与参数名称。
"""只保留 latent mixer 的局部卷积和 FFN 残差路径。"""

from __future__ import annotations

from torch import nn
import torch

from .hybrid_latent_mixer import (
    HybridLatentMixerBlock,
    HybridLatentMixerConfig,
    HybridLatentTemporalMixer,
)


class LocalOnlyLatentMixerBlock(nn.Module):
    """沿用 Hybrid block 的 local/FFN 子模块，不创建或重置参数。"""

    def __init__(self, source: HybridLatentMixerBlock) -> None:
        super().__init__()
        if not isinstance(source, HybridLatentMixerBlock):
            raise TypeError("source 必须是 HybridLatentMixerBlock")
        # 直接移交已初始化子模块，保证其参数值、名称和 RNG 状态不变。
        self.local_norm = source.local_norm
        self.depthwise_conv = source.depthwise_conv
        self.local_expand = source.local_expand
        self.local_activation = source.local_activation
        self.local_project = source.local_project
        self.local_dropout = source.local_dropout
        self.ffn_norm = source.ffn_norm
        self.ffn = source.ffn

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """执行 local convolution 与 FFN 两条残差路径。"""
        local = self.local_norm(value).transpose(1, 2)
        local = self.depthwise_conv(local)
        local = self.local_project(self.local_activation(self.local_expand(local)))
        value = value + self.local_dropout(local.transpose(1, 2))
        return value + self.ffn(self.ffn_norm(value))


class LocalOnlyLatentTemporalMixer(nn.Module):
    """在 ``[B,256,D]`` 上只执行 local convolution 与 FFN。"""

    def __init__(
        self,
        config: HybridLatentMixerConfig,
        blocks: list[LocalOnlyLatentMixerBlock],
    ) -> None:
        super().__init__()
        if len(blocks) != config.depth:
            raise ValueError("blocks 数量必须与 config.depth 一致")
        self.config = config
        self.blocks = nn.ModuleList(blocks)

    @classmethod
    def from_hybrid(
        cls, mixer: HybridLatentTemporalMixer
    ) -> tuple["LocalOnlyLatentTemporalMixer", list[dict[str, int | str]]]:
        """删除 global 子模块并返回其参数名与参数量审计表。"""
        if not isinstance(mixer, HybridLatentTemporalMixer):
            raise TypeError("mixer 必须是 HybridLatentTemporalMixer")

        removed = []
        for name, parameter in mixer.named_parameters():
            if ".global_norm." in name or ".global_attention." in name:
                removed.append({"name": name, "numel": int(parameter.numel())})
        blocks = [LocalOnlyLatentMixerBlock(block) for block in mixer.blocks]
        result = cls(mixer.config, blocks)
        return result, removed

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """按原 mixer 的输入合同逐层应用 local-only block。"""
        expected = (self.config.sequence_length, self.config.hidden_dim)
        if not torch.is_tensor(value) or value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "Local-only mixer 输入必须是 [B,{},{}]，实际为 {}".format(
                    *expected,
                    tuple(value.shape) if torch.is_tensor(value) else type(value),
                )
            )
        for block in self.blocks:
            value = block(value)
        return value


__all__ = ["LocalOnlyLatentMixerBlock", "LocalOnlyLatentTemporalMixer"]
