# SPDX-License-Identifier: GPL-3.0-only
# 从论文研究实现移入；仅调整包内导入和公开说明，保留计算与参数名称。
"""固定 512 帧 BOLD-VAE V0 及其首个 temporal control。

这个模块故意与旧的 ``cfm.vae`` 隔离。V0 只接受固定形状
``[B, 512, 64]``，通过可学习的 stride-2 temporal convolution 压缩时间轴，
再由对称的 learned upsampling 恢复到原始分辨率。默认配置是设计文档中的
V0-A（4x temporal compression）；显式设置 ``temporal_compression=2`` 可运行
设计文档要求的 V0-B 对照（2x temporal compression）。

所有输入信息都必须经过 ``[latent_length, 16]`` latent bottleneck；这里不提供 U-Net
式的高分辨率 skip connection，避免 reconstruction ceiling 依赖后续 CFM
无法生成的额外信息。
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .utils import SinusoidalPositionalEncoding, reparameterize


@dataclass(frozen=True)
class BOLDVAEConfig:
    """BOLD-VAE V0 的模型超参数。

    ``input_length``、``latent_length``、``roi_count``、``latent_dim`` 和
    ``temporal_compression`` 是实验合同。V0-A/V0-B 之间只显式改变
    ``temporal_compression`` 与由它决定的 ``latent_length``，而不是让模型
    自动适应任意长度。
    """

    input_length: int = 512
    latent_length: int = 128
    temporal_compression: int = 4
    roi_count: int = 64
    latent_dim: int = 16
    hidden_width: int = 128
    attention_heads: int = 4
    bottleneck_depth: int = 2
    ffn_dim: int = 256
    dropout: float = 0.1
    beta_kl: float = 0.001
    spectral_weight: float = 0.1
    resblock_depth: int = 1
    tcorr_weight: float = 0.0
    fc_weight: float = 0.0

    def __post_init__(self) -> None:
        positive_fields = (
            "input_length",
            "latent_length",
            "temporal_compression",
            "roi_count",
            "latent_dim",
            "hidden_width",
            "attention_heads",
            "bottleneck_depth",
            "ffn_dim",
            "resblock_depth",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} 必须是正整数".format(name))
        if self.input_length != 512:
            raise ValueError("BOLD-VAE 的 input_length 必须是 512")
        if self.temporal_compression not in (2, 4):
            raise ValueError("首个 temporal ablation 只支持 2x 或 4x compression")
        if self.input_length % self.temporal_compression != 0:
            raise ValueError("input_length 必须能被 temporal_compression 整除")
        if self.latent_length != self.input_length // self.temporal_compression:
            raise ValueError(
                "latent_length 必须等于 input_length / temporal_compression"
            )
        if self.roi_count != 64:
            raise ValueError("BOLD-VAE V0 的 roi_count 必须是 64")
        if self.latent_dim != 16:
            raise ValueError("BOLD-VAE V0 的 latent_dim 必须是 16")
        if self.hidden_width % self.attention_heads != 0:
            raise ValueError("hidden_width 必须能被 attention_heads 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须在 [0, 1) 内")
        if self.beta_kl < 0.0 or self.spectral_weight < 0.0:
            raise ValueError("beta_kl 和 spectral_weight 不能为负数")
        for name in ("tcorr_weight", "fc_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError("{} 必须是有限非负数".format(name))


def _group_count(channels: int) -> int:
    """选择能整除 channel 数的 GroupNorm group 数。"""
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class TemporalResBlock(nn.Module):
    """保持时间长度不变的卷积 residual block。"""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.conv1(F.silu(self.norm1(hidden)))
        hidden = self.dropout(hidden)
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return residual + hidden


class TemporalDownsample(nn.Module):
    """learned 2x temporal downsampling，长度恰好减半。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # kernel=4、stride=2、padding=1 对偶数输入给出严格的 L/2。
        self.projection = nn.Conv1d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden)


class TemporalUpsample(nn.Module):
    """nearest-neighbor 2x 插值后接可学习 Conv1D refine。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        return self.refine(hidden)


class BottleneckTemporalTransformer(nn.Module):
    """只在 compressed-token bottleneck 上做全局 temporal modeling。"""

    def __init__(self, config: BOLDVAEConfig) -> None:
        super().__init__()
        self.position = SinusoidalPositionalEncoding(
            config.hidden_width, config.latent_length
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_width,
            nhead=config.attention_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.bottleneck_depth,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.hidden_width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.position(hidden)
        return self.output_norm(self.encoder(hidden))


class BOLDVAEEncoder(nn.Module):
    """将 ``[B,512,64]`` 编码成 ``mu/logvar: [B,L_latent,16]``。"""

    def __init__(self, config: BOLDVAEConfig) -> None:
        super().__init__()
        width = config.hidden_width
        self.input_projection = nn.Conv1d(
            config.roi_count, width, kernel_size=3, padding=1
        )
        self.input_blocks = nn.Sequential(
            *(
                TemporalResBlock(width, config.dropout)
                for _ in range(config.resblock_depth)
            )
        )
        self.downsample1 = TemporalDownsample(width)
        self.middle_blocks = nn.Sequential(
            *(
                TemporalResBlock(width, config.dropout)
                for _ in range(config.resblock_depth)
            )
        )
        self.downsample2 = (
            TemporalDownsample(width)
            if config.temporal_compression == 4
            else nn.Identity()
        )
        self.bottleneck_blocks = nn.Sequential(
            *(
                TemporalResBlock(width, config.dropout)
                for _ in range(config.resblock_depth)
            )
        )
        self.temporal_model = BottleneckTemporalTransformer(config)
        self.mean_head = nn.Linear(width, config.latent_dim)
        self.logvar_head = nn.Linear(width, config.latent_dim)

    def forward(self, bold: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = bold.transpose(1, 2)  # [B, 64, 512]，Conv1D 使用 C-first。
        hidden = self.input_blocks(self.input_projection(hidden))
        hidden = self.middle_blocks(self.downsample1(hidden))
        hidden = self.downsample2(hidden)
        hidden = self.bottleneck_blocks(hidden)
        hidden = hidden.transpose(1, 2)  # [B, latent_length, W]
        hidden = self.temporal_model(hidden)
        return self.mean_head(hidden), self.logvar_head(hidden)


class BOLDVAEDecoder(nn.Module):
    """将 ``[B,L_latent,16]`` 解码成 ``[B,512,64]``。"""

    def __init__(self, config: BOLDVAEConfig) -> None:
        super().__init__()
        width = config.hidden_width
        self.latent_projection = nn.Linear(config.latent_dim, width)
        self.temporal_model = BottleneckTemporalTransformer(config)
        self.bottleneck_blocks = nn.Sequential(
            *(
                TemporalResBlock(width, config.dropout)
                for _ in range(config.resblock_depth)
            )
        )
        self.upsample1 = TemporalUpsample(width)
        self.middle_blocks = nn.Sequential(
            *(
                TemporalResBlock(width, config.dropout)
                for _ in range(config.resblock_depth)
            )
        )
        self.upsample2 = (
            TemporalUpsample(width)
            if config.temporal_compression == 4
            else nn.Identity()
        )
        self.output_blocks = nn.Sequential(
            *(
                TemporalResBlock(width, config.dropout)
                for _ in range(config.resblock_depth)
            )
        )
        self.output_projection = nn.Conv1d(
            width, config.roi_count, kernel_size=3, padding=1
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.temporal_model(self.latent_projection(latent))
        hidden = self.bottleneck_blocks(hidden.transpose(1, 2))
        hidden = self.middle_blocks(self.upsample1(hidden))
        hidden = self.output_blocks(self.upsample2(hidden))
        return self.output_projection(hidden).transpose(1, 2)


def spectral_log_magnitude_loss(
    reconstruction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """沿时间轴、逐 ROI 比较 log-rFFT magnitude。

    输入顺序是 ``[B,T,ROI]``，所以 FFT 明确使用 ``dim=1``；ROI 维度绝不
    参与 FFT。固定长度 V0 不需要 padding mask。
    """
    if reconstruction.shape != target.shape or reconstruction.ndim != 3:
        raise ValueError("reconstruction 和 target 必须同为 [B,T,ROI]")
    reconstruction_spectrum = torch.log1p(
        torch.fft.rfft(reconstruction, dim=1).abs()
    )
    target_spectrum = torch.log1p(torch.fft.rfft(target, dim=1).abs())
    return F.mse_loss(reconstruction_spectrum, target_spectrum)


def temporal_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """沿时间轴逐 ROI 计算 ``1 - Pearson``，再对 batch/ROI 求平均。

    结构损失固定在 FP32 中计算；对零方差轨迹使用 ``eps`` 稳定分母，保证
    常数窗口仍能产生有限 loss 和有限梯度。
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction/target 必须同为 [B,T,ROI]")
    if prediction.shape[1] < 2:
        raise ValueError("TCorr loss 至少需要两个时间点")
    prediction = prediction.float()
    target = target.float()
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction * target).sum(dim=1)
    denominator = torch.sqrt(
        prediction.square().sum(dim=1).clamp_min(eps)
        * target.square().sum(dim=1).clamp_min(eps)
    )
    correlation = numerator / denominator
    return (1.0 - correlation).mean()


def functional_connectivity_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """比较每个窗口 ROI Pearson FC 矩阵 strict upper triangle 的 MSE。

    FC edge 的 Pearson loss 刻意不参与首个 pilot；这里仅保留 correlation
    matrix 的 MSE，并在 FP32 中对常数 ROI 使用稳定分母。
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction/target 必须同为 [B,T,ROI]")
    if prediction.shape[1] < 2:
        raise ValueError("FC loss 至少需要两个时间点")
    if prediction.shape[2] < 2:
        raise ValueError("FC loss 至少需要两个 ROI")
    prediction = prediction.float()
    target = target.float()

    def correlation_matrix(sequence: torch.Tensor) -> torch.Tensor:
        centered = sequence - sequence.mean(dim=1, keepdim=True)
        squared_norm = centered.square().sum(dim=1)
        denominator = torch.sqrt(
            squared_norm.clamp_min(eps).unsqueeze(2)
            * squared_norm.clamp_min(eps).unsqueeze(1)
        )
        numerator = centered.transpose(1, 2) @ centered
        return numerator / denominator

    prediction_fc = correlation_matrix(prediction)
    target_fc = correlation_matrix(target)
    upper = torch.triu_indices(
        prediction.shape[2], prediction.shape[2], offset=1, device=prediction.device
    )
    return F.mse_loss(
        prediction_fc[:, upper[0], upper[1]], target_fc[:, upper[0], upper[1]]
    )


def gaussian_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """标准 diagonal Gaussian 到 N(0,I) 的 KL，平均 batch/time/dim。"""
    if mu.shape != logvar.shape or mu.ndim != 3:
        raise ValueError("mu 和 logvar 必须同为 [B,T,D]")
    elementwise = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return elementwise.mean()


@dataclass
class BOLDVAEOutput:
    """V0 forward 的完整结果与分项 loss。"""

    reconstruction: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    spectral_loss: torch.Tensor
    tcorr_loss: torch.Tensor
    fc_loss: torch.Tensor
    total_loss: torch.Tensor

    @property
    def mean(self) -> torch.Tensor:
        """兼容旧代码的命名；新的 API 使用 ``mu``。"""
        return self.mu

    @property
    def loss(self) -> torch.Tensor:
        """兼容旧训练脚本的命名；新的 API 使用 ``total_loss``。"""
        return self.total_loss


class BOLDVAE(nn.Module):
    """固定形状 BOLD-VAE V0-A/V0-B。

    Public contract:

    ``[B,512,64] -> [B,L_latent,16] -> [B,512,64]``，其中 ``L_latent``
    由 ``temporal_compression`` 显式决定。

    ``encode(..., sample=False)`` 返回 posterior mean；训练时 ``forward``
    根据 ``self.training`` 进行 reparameterization，eval 时自动使用 mean。
    """

    def __init__(self, config: Optional[BOLDVAEConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = BOLDVAEConfig(**kwargs)
        elif kwargs:
            raise TypeError("传入 config 后不能再传模型关键字参数")
        if not isinstance(config, BOLDVAEConfig):
            raise TypeError("config 必须是 BOLDVAEConfig")
        self.config = config
        self.encoder = BOLDVAEEncoder(config)
        self.decoder = BOLDVAEDecoder(config)

    def _check_bold(self, bold: torch.Tensor) -> None:
        expected = (self.config.input_length, self.config.roi_count)
        if not torch.is_tensor(bold) or bold.ndim != 3:
            raise ValueError(
                "BOLD 输入必须是 [B,{},{}] tensor".format(*expected)
            )
        if tuple(bold.shape[1:]) != expected:
            raise ValueError(
                "BOLD 输入必须是 [B,{},{}]，实际为 {}".format(
                    *expected, tuple(bold.shape)
                )
            )

    def _check_latent(self, latent: torch.Tensor) -> None:
        expected = (self.config.latent_length, self.config.latent_dim)
        if not torch.is_tensor(latent) or latent.ndim != 3:
            raise ValueError(
                "latent 输入必须是 [B,{},{}] tensor".format(*expected)
            )
        if tuple(latent.shape[1:]) != expected:
            raise ValueError(
                "latent 输入必须是 [B,{},{}]，实际为 {}".format(
                    *expected, tuple(latent.shape)
                )
            )

    def encode_distribution(
        self, bold: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 posterior ``mu`` 与 ``logvar``，形状均为 ``[B,L_latent,16]``。"""
        self._check_bold(bold)
        return self.encoder(bold)

    def encode(self, bold: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """编码 BOLD；默认返回 deterministic posterior mean。"""
        mu, logvar = self.encode_distribution(bold)
        return reparameterize(mu, logvar, sample=sample)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码 latent，输出严格为 ``[B,512,64]``。"""
        self._check_latent(latent)
        return self.decoder(latent)

    def forward(self, bold: torch.Tensor) -> BOLDVAEOutput:
        """执行 VAE forward，并返回基础与结构 loss 分项。"""
        self._check_bold(bold)
        mu, logvar = self.encoder(bold)
        z = reparameterize(mu, logvar, sample=self.training)
        reconstruction = self.decode(z)
        recon_loss = F.mse_loss(reconstruction, bold)
        kl_loss = gaussian_kl_loss(mu, logvar)
        spectral_loss = spectral_log_magnitude_loss(reconstruction, bold)
        tcorr_loss = temporal_correlation_loss(reconstruction, bold)
        fc_loss = functional_connectivity_mse_loss(reconstruction, bold)
        legacy_total_loss = (
            recon_loss
            + self.config.beta_kl * kl_loss
            + self.config.spectral_weight * spectral_loss
        )
        if self.config.tcorr_weight == 0.0 and self.config.fc_weight == 0.0:
            # 默认 V0 objective 保持与旧路径逐 bit 相同。
            total_loss = legacy_total_loss
        else:
            total_loss = (
                legacy_total_loss
                + self.config.tcorr_weight * tcorr_loss
                + self.config.fc_weight * fc_loss
            )
        return BOLDVAEOutput(
            reconstruction=reconstruction,
            z=z,
            mu=mu,
            logvar=logvar,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            spectral_loss=spectral_loss,
            tcorr_loss=tcorr_loss,
            fc_loss=fc_loss,
            total_loss=total_loss,
        )


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数量，供 checkpoint/实验 handoff 使用。"""
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


__all__ = [
    "BOLDVAEConfig",
    "BOLDVAEEncoder",
    "BOLDVAEDecoder",
    "BOLDVAEOutput",
    "BOLDVAE",
    "count_parameters",
    "gaussian_kl_loss",
    "functional_connectivity_mse_loss",
    "spectral_log_magnitude_loss",
    "temporal_correlation_loss",
]
