# SPDX-License-Identifier: GPL-3.0-only
# 从论文研究实现移入；仅调整包内导入和公开说明，保留计算与参数名称。
"""Hybrid V1 的 nonlinear BOLD-VAE latent projection head。"""

from __future__ import annotations

import torch
from torch import nn


class HybridLatentProjectionHead(nn.Module):
    """``D -> D -> 16`` 的无界 latent regression head。"""

    def __init__(
        self,
        hidden_dim: int = 256,
        latent_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or latent_dim <= 0:
            raise ValueError("hidden_dim/latent_dim 必须是正整数")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output_projection(
            self.dropout(self.activation(self.hidden_projection(self.norm(value))))
        )


__all__ = ["HybridLatentProjectionHead"]
