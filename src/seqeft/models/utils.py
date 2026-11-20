# SPDX-License-Identifier: GPL-3.0-only
# 从论文研究实现移入；仅调整包内导入和公开说明，保留计算与参数名称。
"""共享工具：masked loss、位置/时间编码与 VAE 重参数化。"""

import math

import torch
from torch import nn


def masked_mse(prediction, target, length):
    """只对每个样本前 ``length`` 个时间 token 计算 MSE，忽略 padding。"""
    if prediction.shape != target.shape:
        raise ValueError("prediction 与 target 形状必须相同")
    if prediction.ndim < 2:
        raise ValueError("至少需要 [B, L, ...] 形状")
    batch_size, max_length = prediction.shape[:2]
    if tuple(length.shape) != (batch_size,):
        raise ValueError("length 必须是一维且长度等于 batch size")
    mask = torch.arange(max_length, device=prediction.device) < length[:, None]
    mask = mask.view(batch_size, max_length, *([1] * (prediction.ndim - 2)))
    error = (prediction - target).pow(2)
    scalar_per_token = 1
    for size in prediction.shape[2:]:
        scalar_per_token *= size
    denominator = float(mask.sum().item()) * scalar_per_token
    return (error * mask.to(prediction.dtype)).sum() / denominator


def kl_loss(mean, logvar, length):
    """标准高斯 KL，按有效时间 token 平均。"""
    if mean.shape != logvar.shape or mean.ndim < 2:
        raise ValueError("mean/logvar 必须形状相同且至少为 [B, L, D]")
    batch_size, max_length = mean.shape[:2]
    if tuple(length.shape) != (batch_size,):
        raise ValueError("length 必须是一维且长度等于 batch size")
    kl = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp())
    mask = torch.arange(max_length, device=mean.device) < length[:, None]
    mask = mask.view(batch_size, max_length, *([1] * (kl.ndim - 2)))
    scalar_per_token = 1
    for size in kl.shape[2:]:
        scalar_per_token *= size
    denominator = float(mask.sum().item()) * scalar_per_token
    return (kl * mask.to(kl.dtype)).sum() / denominator


def reparameterize(mean, logvar, sample=True):
    """VAE 重参数化；``sample=False`` 时返回均值，供 Flow 作条件/target。"""
    if not sample:
        return mean
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return mean + epsilon * std


class SinusoidalPositionalEncoding(nn.Module):
    """动态 sinusoidal 位置编码，任意长度直接计算，不受 max_len 限制。"""

    def __init__(self, width: int, max_len: int = 4096) -> None:
        super().__init__()
        # max_len 保留为接口兼容参数；forward 按实际长度动态生成位置表。
        self.width = width
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).unsqueeze(1).float()
        divisor = torch.exp(
            torch.arange(0, self.width, 2, device=tokens.device).float()
            * (-math.log(10000.0) / self.width)
        )
        table = torch.zeros(length, self.width, device=tokens.device)
        table[:, 0::2] = torch.sin(positions * divisor)
        table[:, 1::2] = torch.cos(positions * divisor)
        return tokens + table.unsqueeze(0)


class SinusoidalTimeEmbedding(nn.Module):
    """Flow time 的 sinusoidal embedding。"""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width % 2 != 0:
            raise ValueError("width 必须是偶数")
        self.half = width // 2
        self.width = width

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        divisor = torch.exp(
            torch.arange(self.half, device=time.device).float()
            * (-math.log(10000.0) / self.half)
        )
        args = time[:, None] * divisor[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
