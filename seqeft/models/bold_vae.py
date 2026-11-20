# SPDX-License-Identifier: GPL-3.0-only
"""Sequence VAE that compresses and reconstructs 512-frame BOLD windows."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class BOLDVAEConfig:
    """VAE architecture and loss settings for 2x or 4x temporal compression."""

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
                raise ValueError("{} must be a positive integer".format(name))
        if self.input_length != 512:
            raise ValueError("input_length must be 512")
        if self.temporal_compression not in (2, 4):
            raise ValueError("temporal_compression must be 2 or 4")
        if self.input_length % self.temporal_compression != 0:
            raise ValueError("input_length must be divisible by temporal_compression")
        if self.latent_length != self.input_length // self.temporal_compression:
            raise ValueError(
                "latent_length must equal input_length / temporal_compression"
            )
        if self.roi_count != 64:
            raise ValueError("roi_count must be 64")
        if self.latent_dim != 16:
            raise ValueError("latent_dim must be 16")
        if self.hidden_width % self.attention_heads != 0:
            raise ValueError("hidden_width must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.beta_kl < 0.0 or self.spectral_weight < 0.0:
            raise ValueError("beta_kl and spectral_weight must be nonnegative")
        for name in ("tcorr_weight", "fc_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError("{} must be finite and nonnegative".format(name))


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class TemporalResBlock(nn.Module):
    """Convolutional residual block that preserves sequence length."""

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
    """Learned downsampling by a factor of two."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv1d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden)


class TemporalUpsample(nn.Module):
    """Upsample by two with nearest-neighbor interpolation and a 1D convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        return self.refine(hidden)


class SinusoidalPositionalEncoding(nn.Module):
    """Compute sinusoidal position encodings for the input length."""

    def __init__(self, width: int, max_len: int = 4096) -> None:
        super().__init__()
        # Retain max_len for compatibility with existing callers.
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


class BottleneckTemporalTransformer(nn.Module):
    """Apply temporal self-attention at the compressed sequence length."""

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
    """Encode [B, 512, 64] as a posterior mean and log variance of shape [B, L, 16]."""

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
        hidden = bold.transpose(1, 2)
        hidden = self.input_blocks(self.input_projection(hidden))
        hidden = self.middle_blocks(self.downsample1(hidden))
        hidden = self.downsample2(hidden)
        hidden = self.bottleneck_blocks(hidden)
        hidden = hidden.transpose(1, 2)  # [B, latent_length, W]
        hidden = self.temporal_model(hidden)
        return self.mean_head(hidden), self.logvar_head(hidden)


class BOLDVAEDecoder(nn.Module):
    """Decode [B, L, 16] latents to [B, 512, 64] BOLD."""

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
    """Compare log FFT magnitudes along the time axis for each ROI."""
    if reconstruction.shape != target.shape or reconstruction.ndim != 3:
        raise ValueError("Reconstruction and target must share shape [B, T, ROI]")
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
    """Average one minus temporal Pearson correlation across batches and ROIs."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Prediction and target must share shape [B, T, ROI]")
    if prediction.shape[1] < 2:
        raise ValueError("Temporal correlation requires at least two frames")
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
    """Compute MSE between the strict upper triangles of ROI correlation matrices."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Prediction and target must share shape [B, T, ROI]")
    if prediction.shape[1] < 2:
        raise ValueError("FC loss requires at least two frames")
    if prediction.shape[2] < 2:
        raise ValueError("FC loss requires at least two ROIs")
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


def reparameterize(mean, logvar, sample=True):
    """Sample a diagonal Gaussian by reparameterization or return its mean."""
    if not sample:
        return mean
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return mean + epsilon * std


def gaussian_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute mean KL divergence from the diagonal posterior to a standard normal."""
    if mu.shape != logvar.shape or mu.ndim != 3:
        raise ValueError("mu and logvar must share shape [B, T, D]")
    elementwise = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return elementwise.mean()


@dataclass
class BOLDVAEOutput:
    """Reconstruction, latents, and loss terms."""

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
        """Alias for mu."""
        return self.mu

    @property
    def loss(self) -> torch.Tensor:
        """Alias for total_loss."""
        return self.total_loss


class BOLDVAE(nn.Module):
    """Reconstruct BOLD through a temporal bottleneck with 16 features per step."""

    def __init__(self, config: Optional[BOLDVAEConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = BOLDVAEConfig(**kwargs)
        elif kwargs:
            raise TypeError("Pass either config or keyword settings")
        if not isinstance(config, BOLDVAEConfig):
            raise TypeError("config must be a BOLDVAEConfig")
        self.config = config
        self.encoder = BOLDVAEEncoder(config)
        self.decoder = BOLDVAEDecoder(config)

    def _check_bold(self, bold: torch.Tensor) -> None:
        expected = (self.config.input_length, self.config.roi_count)
        if not torch.is_tensor(bold) or bold.ndim != 3:
            raise ValueError(
                "BOLD must be a tensor with shape [B, {}, {}]".format(*expected)
            )
        if tuple(bold.shape[1:]) != expected:
            raise ValueError(
                "Expected BOLD shape [B, {}, {}], got {}".format(
                    *expected, tuple(bold.shape)
                )
            )

    def _check_latent(self, latent: torch.Tensor) -> None:
        expected = (self.config.latent_length, self.config.latent_dim)
        if not torch.is_tensor(latent) or latent.ndim != 3:
            raise ValueError(
                "Latent input must be a tensor with shape [B, {}, {}]".format(*expected)
            )
        if tuple(latent.shape[1:]) != expected:
            raise ValueError(
                "Expected latent shape [B, {}, {}], got {}".format(
                    *expected, tuple(latent.shape)
                )
            )

    def encode_distribution(
        self, bold: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log variance."""
        self._check_bold(bold)
        return self.encoder(bold)

    def encode(self, bold: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Encode BOLD, using the posterior mean by default."""
        mu, logvar = self.encode_distribution(bold)
        return reparameterize(mu, logvar, sample=sample)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self._check_latent(latent)
        return self.decoder(latent)

    def forward(self, bold: torch.Tensor) -> BOLDVAEOutput:
        """Sample the posterior during training and use its mean during evaluation."""
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
            # Keep the reference loss arithmetic when the optional terms are disabled.
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
    """Count all parameters or just the trainable ones."""
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
