# SPDX-License-Identifier: GPL-3.0-only
"""Compress each frame's LaBraM and spectral tokens into learned EEG slots."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .labram import (
    CheckpointLoadReport,
    LaBraMTemporalEncoder,
    load_temporal_checkpoint,
)
from .spectral import MultiScaleSpectralEncoder


@dataclass(frozen=True)
class HybridTokenizerConfig:

    hidden_dim: int = 256
    input_scale: float = 0.01
    frame_chunk_size: int = 8
    slots_per_branch: int = 8
    depth: int = 2
    heads: int = 8
    dropout: float = 0.1
    sequence_length: int = 512
    channels: int = 26
    patches: int = 16
    samples_per_patch: int = 200
    temporal_token_count: int = 416
    spectral_token_count: int = 832

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim",
            "frame_chunk_size",
            "slots_per_branch",
            "depth",
            "heads",
            "sequence_length",
            "channels",
            "patches",
            "samples_per_patch",
            "temporal_token_count",
            "spectral_token_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if self.sequence_length != 512:
            raise ValueError("sequence_length must be 512")
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.input_scale <= 0.0:
            raise ValueError("input_scale must be positive")


class _PerceiverTokenizerBlock(nn.Module):
    """Cross-attention to raw tokens, followed by slot self-attention and an FFN."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.raw_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(dropout)

        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, latent: torch.Tensor, raw_tokens: torch.Tensor) -> torch.Tensor:
        raw = self.raw_norm(raw_tokens)
        query = self.cross_norm(latent)
        attended = self.cross_attention(
            query,
            raw,
            raw,
            need_weights=False,
        )[0]
        latent = latent + self.cross_dropout(attended)

        query = self.self_norm(latent)
        attended = self.self_attention(
            query,
            query,
            query,
            need_weights=False,
        )[0]
        latent = latent + self.self_dropout(attended)
        return latent + self.ffn(self.ffn_norm(latent))


class HybridEEGTokenizer(nn.Module):
    """Map [B, 512, 26, 16, 200] EEG to [B, 512, slots, hidden_dim]."""

    def __init__(
        self,
        temporal: Optional[nn.Module] = None,
        spectral: Optional[nn.Module] = None,
        config: Optional[HybridTokenizerConfig] = None,
        *,
        freeze_temporal: bool = False,
        **config_kwargs,
    ) -> None:
        super().__init__()
        if config is not None and config_kwargs:
            raise TypeError("Pass either config or tokenizer keyword settings")
        self.config = config or HybridTokenizerConfig(**config_kwargs)
        self.temporal = temporal if temporal is not None else LaBraMTemporalEncoder()
        self.spectral = spectral if spectral is not None else MultiScaleSpectralEncoder()
        if not hasattr(self.temporal, "forward_patch_tokens"):
            raise TypeError("temporal must provide forward_patch_tokens()")
        if not hasattr(self.spectral, "forward_tokens"):
            raise TypeError("spectral must provide forward_tokens()")

        width = self.config.hidden_dim
        feature_dim = self.config.samples_per_patch
        self.temporal_input_norm = nn.LayerNorm(feature_dim)
        self.temporal_projection = nn.Linear(feature_dim, width)
        self.spectral_input_norm = nn.LayerNorm(feature_dim)
        self.spectral_projection = nn.Linear(feature_dim, width)

        temporal_slots = torch.empty(1, self.config.slots_per_branch, width)
        spectral_slots = torch.empty(1, self.config.slots_per_branch, width)
        self.temporal_latent_slots = nn.Parameter(temporal_slots)
        self.spectral_latent_slots = nn.Parameter(spectral_slots)
        nn.init.trunc_normal_(self.temporal_latent_slots, std=0.02)
        nn.init.trunc_normal_(self.spectral_latent_slots, std=0.02)

        self.temporal_tokenizer = nn.ModuleList(
            _PerceiverTokenizerBlock(width, self.config.heads, self.config.dropout)
            for _ in range(self.config.depth)
        )
        self.spectral_tokenizer = nn.ModuleList(
            _PerceiverTokenizerBlock(width, self.config.heads, self.config.dropout)
            for _ in range(self.config.depth)
        )

        total_slots = self.config.slots_per_branch * 2
        self.type_embedding = nn.Embedding(2, width)
        self.slot_embedding = nn.Embedding(total_slots, width)
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=self.config.heads,
            dim_feedforward=width * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.frame_fusion = nn.TransformerEncoder(
            fusion_layer,
            num_layers=1,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.frame_position = nn.Parameter(
            torch.empty(1, self.config.sequence_length, width)
        )
        nn.init.trunc_normal_(self.type_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.slot_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.frame_position, std=0.02)

        self.freeze_temporal = bool(freeze_temporal)
        self.load_report: Optional[CheckpointLoadReport] = None
        if self.freeze_temporal:
            self.temporal.requires_grad_(False)
            self.temporal.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        config: Optional[HybridTokenizerConfig] = None,
        *,
        freeze_temporal: bool = False,
        spectral_seed: int = 2026,
        **config_kwargs,
    ) -> "HybridEEGTokenizer":
        """Load the LaBraM encoder and seed the spectral encoder separately."""
        if config is not None and config_kwargs:
            raise TypeError("Pass either config or tokenizer keyword settings")
        tokenizer_config = config or HybridTokenizerConfig(**config_kwargs)
        temporal = LaBraMTemporalEncoder()
        report = load_temporal_checkpoint(temporal, Path(checkpoint_path))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(spectral_seed))
            spectral = MultiScaleSpectralEncoder()
        tokenizer = cls(
            temporal=temporal,
            spectral=spectral,
            config=tokenizer_config,
            freeze_temporal=freeze_temporal,
        )
        tokenizer.load_report = report
        return tokenizer

    def train(self, mode: bool = True) -> "HybridEEGTokenizer":
        """Keep the temporal encoder in evaluation mode when frozen."""
        super().train(mode)
        if self.freeze_temporal:
            self.temporal.eval()
        return self

    @staticmethod
    def _validate_eeg(eeg: torch.Tensor) -> None:
        expected = (512, 26, 16, 200)
        if not torch.is_tensor(eeg) or eeg.ndim != 5 or tuple(eeg.shape[1:]) != expected:
            raise ValueError(
                "Expected EEG shape [B, 512, 26, 16, 200], got {}".format(
                    tuple(eeg.shape) if torch.is_tensor(eeg) else type(eeg)
                )
            )

    def _run_temporal(self, frames: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_temporal else nullcontext()
        with context:
            tokens = self.temporal.forward_patch_tokens(frames)
        expected = (frames.shape[0], self.config.temporal_token_count, 200)
        if not torch.is_tensor(tokens) or tuple(tokens.shape) != expected:
            raise RuntimeError(
                "Expected LaBraM token shape {}, got {}".format(
                    expected,
                    tuple(tokens.shape) if torch.is_tensor(tokens) else type(tokens),
                )
            )
        return tokens

    def _run_spectral(self, frames: torch.Tensor) -> torch.Tensor:
        tokens = self.spectral.forward_tokens(
            frames.reshape(frames.shape[0], 26, 16 * 200)
        )
        expected = (frames.shape[0], self.config.spectral_token_count, 200)
        if not torch.is_tensor(tokens) or tuple(tokens.shape) != expected:
            raise RuntimeError(
                "Expected spectral token shape {}, got {}".format(
                    expected,
                    tuple(tokens.shape) if torch.is_tensor(tokens) else type(tokens),
                )
            )
        return tokens

    @staticmethod
    def _tokenize_branch(
        tokens: torch.Tensor,
        projection_norm: nn.LayerNorm,
        projection: nn.Linear,
        slots: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        projected = projection(projection_norm(tokens))
        latent = slots.expand(projected.shape[0], -1, -1)
        for block in blocks:
            latent = block(latent, projected)
        return latent

    def _fuse_frame_slots(
        self,
        temporal_slots: torch.Tensor,
        spectral_slots: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.cat((temporal_slots, spectral_slots), dim=1)
        device = value.device
        type_ids = torch.cat(
            (
                torch.zeros(self.config.slots_per_branch, dtype=torch.long, device=device),
                torch.ones(self.config.slots_per_branch, dtype=torch.long, device=device),
            )
        )
        slot_ids = torch.arange(value.shape[1], dtype=torch.long, device=device)
        value = value + self.type_embedding(type_ids).unsqueeze(0)
        value = value + self.slot_embedding(slot_ids).unsqueeze(0)
        return self.frame_fusion(value)

    def _forward_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        temporal_tokens = self._run_temporal(chunk)
        spectral_tokens = self._run_spectral(chunk)
        temporal_slots = self._tokenize_branch(
            temporal_tokens,
            self.temporal_input_norm,
            self.temporal_projection,
            self.temporal_latent_slots,
            self.temporal_tokenizer,
        )
        spectral_slots = self._tokenize_branch(
            spectral_tokens,
            self.spectral_input_norm,
            self.spectral_projection,
            self.spectral_latent_slots,
            self.spectral_tokenizer,
        )
        return self._fuse_frame_slots(temporal_slots, spectral_slots)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        self._validate_eeg(eeg)
        batch, frames = eeg.shape[:2]
        flattened = (eeg * self.config.input_scale).reshape(
            batch * frames,
            self.config.channels,
            self.config.patches,
            self.config.samples_per_patch,
        )
        outputs = []
        chunk_size = self.config.frame_chunk_size
        for start in range(0, flattened.shape[0], chunk_size):
            chunk = flattened[start : start + chunk_size]
            # Recompute chunk activations during backward to bound GPU memory.
            if self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self._forward_chunk, chunk, use_reentrant=False))
            else:
                outputs.append(self._forward_chunk(chunk))

        value = torch.cat(outputs, dim=0).reshape(
            batch,
            frames,
            self.config.slots_per_branch * 2,
            self.config.hidden_dim,
        )
        return value + self.frame_position[:, :frames].unsqueeze(2)

    def provenance(self) -> dict:
        """Report tokenizer settings and the LaBraM checkpoint hash."""
        return {
            "input_scale": self.config.input_scale,
            "frame_chunk_size": self.config.frame_chunk_size,
            "slots_per_branch": self.config.slots_per_branch,
            "tokenizer_depth": self.config.depth,
            "freeze_temporal": self.freeze_temporal,
            "labram_checkpoint_sha256": (
                self.load_report.checkpoint_sha256
                if self.load_report is not None
                else None
            ),
        }


__all__ = ["HybridEEGTokenizer", "HybridTokenizerConfig"]
