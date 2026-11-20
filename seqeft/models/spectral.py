# SPDX-License-Identifier: GPL-3.0-only
"""Encode 16-second EEG windows with four STFT scales and linear attention."""

import math
import warnings

import torch
from linear_attention_transformer import LinearAttentionTransformer
from torch import nn


class PatchFrequencyEmbedding(nn.Module):
    """Project spectral bins to the model width."""

    def __init__(self, frequency_bins: int, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(frequency_bins, width)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.projection(spectrum.permute(0, 2, 1))


class PatchWindowEmbedding(nn.Module):
    """Project the time-window axis to 32 tokens."""

    def __init__(self, source_windows: int, target_windows: int = 32) -> None:
        super().__init__()
        self.projection = nn.Linear(source_windows, target_windows)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = self.projection(tokens.permute(0, 2, 1))
        return value.permute(0, 2, 1)


class SinusoidalPositionEncoding(nn.Module):
    """Apply fixed sinusoidal positions and dropout."""

    def __init__(
        self,
        width: int,
        dropout: float = 0.1,
        max_length: int = 1000,
    ) -> None:
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1).float()
        divisor = torch.exp(
            torch.arange(0, width, 2).float()
            * -(math.log(10000.0) / width)
        )
        table = torch.zeros(max_length, width)
        table[:, 0::2] = torch.sin(positions * divisor)
        table[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("table", table.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(tokens + self.table[:, : tokens.shape[1]])


class MultiScaleSpectralEncoder(nn.Module):
    """Encode 26 EEG channels as 200-dimensional spectral tokens."""

    FFT_SCALES = (100, 200, 400, 800)
    SOURCE_WINDOWS = (32, 16, 8, 4)

    def __init__(
        self,
        *,
        width: int = 200,
        channels: int = 26,
        input_length: int = 3200,
    ) -> None:
        super().__init__()
        if (width, channels, input_length) != (200, 26, 3200):
            raise ValueError("The spectral encoder requires width=200, channels=26, and input_length=3200")
        self.width = width
        self.channels = channels
        self.input_length = input_length
        self.frequency_embeddings = nn.ModuleList([
            PatchFrequencyEmbedding(n_fft // 2 + 1, width)
            for n_fft in self.FFT_SCALES
        ])
        self.window_embeddings = nn.ModuleList([
            PatchWindowEmbedding(source_windows, target_windows=32)
            for source_windows in self.SOURCE_WINDOWS
        ])
        self.channel_tokens = nn.Embedding(channels, width)
        self.register_buffer(
            "channel_indices",
            torch.arange(channels, dtype=torch.long),
        )
        self.position_encoding = SinusoidalPositionEncoding(width, dropout=0.1)
        self.transformer = LinearAttentionTransformer(
            dim=width,
            heads=8,
            depth=4,
            max_seq_len=1024,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @staticmethod
    def _stft_magnitude(sample: torch.Tensor, n_fft: int) -> torch.Tensor:
        if sample.ndim != 3 or sample.shape[1] != 1:
            raise ValueError("STFT input must have shape [batch, 1, time]")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A window was not provided.*",
                category=UserWarning,
            )
            spectrum = torch.stft(
                sample.squeeze(1),
                n_fft=n_fft,
                hop_length=n_fft,
                window=None,
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        return spectrum.abs()

    def forward_tokens(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return [NB, 832, 200] features before pooling."""
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (
            self.channels,
            self.input_length,
        ):
            raise ValueError(
                "Expected spectral encoder input shape [NB, 26, 3200], got {}".format(
                    tuple(eeg.shape)
                )
            )
        batch = eeg.shape[0]
        channel_batch = eeg.reshape(batch * self.channels, 1, self.input_length)
        scale_tokens = []
        for n_fft, frequency_embedding, window_embedding in zip(
            self.FFT_SCALES,
            self.frequency_embeddings,
            self.window_embeddings,
        ):
            spectrum = self._stft_magnitude(channel_batch, n_fft)
            scale_tokens.append(
                window_embedding(frequency_embedding(spectrum))
            )
        tokens = torch.stack(scale_tokens, dim=0).sum(dim=0)
        tokens = tokens.reshape(batch, self.channels, 32, self.width)
        channel_tokens = self.channel_tokens(self.channel_indices).reshape(
            1,
            self.channels,
            1,
            self.width,
        )
        tokens = tokens + channel_tokens
        tokens = self.position_encoding(
            tokens.reshape(batch * self.channels, 32, self.width)
        ).reshape(batch, self.channels * 32, self.width)
        return self.transformer(tokens)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return a mean-pooled spectral feature for each window."""
        return self.forward_tokens(eeg).mean(dim=1)
