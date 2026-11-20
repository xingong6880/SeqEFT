# SPDX-License-Identifier: GPL-3.0-only
# 从论文研究实现移入；仅调整包内导入和公开说明，保留计算与参数名称。
"""SeqEFT 的 multi-scale spectral EEG encoder。

结构同步 NeuroBOLT 的 MSS：对 16 秒 EEG 使用四尺度 STFT，将每个尺度统一为
32 个 200 维 tokens，再用 linear-attention 汇总 26 个通道的 832 个 tokens。
"""

import math
import warnings

import torch
from linear_attention_transformer import LinearAttentionTransformer
from torch import nn


class PatchFrequencyEmbedding(nn.Module):
    """把一个 STFT window 的 frequency bins 投影到 feature width。"""

    def __init__(self, frequency_bins: int, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(frequency_bins, width)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.projection(spectrum.permute(0, 2, 1))


class PatchWindowEmbedding(nn.Module):
    """把不同尺度的 STFT window 数统一投影成 32 个 tokens。"""

    def __init__(self, source_windows: int, target_windows: int = 32) -> None:
        super().__init__()
        self.projection = nn.Linear(source_windows, target_windows)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = self.projection(tokens.permute(0, 2, 1))
        return value.permute(0, 2, 1)


class SinusoidalPositionEncoding(nn.Module):
    """固定 sinusoidal position，加法后应用与 MSS 一致的 dropout。"""

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
    """四尺度 STFT + linear attention，输出一个 200 维 window feature。"""

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
            raise ValueError("本轮 MSS 固定 width/channels/input_length=200/26/3200")
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
            raise ValueError("STFT sample 必须是 [batch,1,time]")
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
        """返回 MSS transformer 后的逐 token 特征 ``[NB,832,200]``。

        新版 EEG condition encoder 需要在任务相关 pooling 之前保留
        channel/window 结构；旧的 ``forward()`` 仍然通过这里的结果做均值，
        因而不会改变历史调用方的 ``[NB,200]`` 输出合同。
        """
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (
            self.channels,
            self.input_length,
        ):
            raise ValueError(
                "MSS 输入必须是 [NB,26,3200]，实际为 {}".format(
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
        """返回 legacy mean-pooled MSS feature ``[NB,200]``。"""
        return self.forward_tokens(eeg).mean(dim=1)
