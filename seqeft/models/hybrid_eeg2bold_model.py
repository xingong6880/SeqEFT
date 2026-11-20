# SPDX-License-Identifier: GPL-3.0-only
"""EEG-to-BOLD model with latent sequence alignment."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .bold_vae import BOLDVAEConfig
from .eeg2bold_model import (
    EEG2BOLDOutput,
    _freeze_eval,
    _validate_bold_vae_config,
    load_frozen_bold_vae,
)
from .hybrid_eeg_tokenizer import HybridEEGTokenizer, HybridTokenizerConfig
from .hybrid_grounding import (
    HybridGroundingConfig,
    OverlappingLocalCrossAttention,
)
from .hybrid_latent_mixer import HybridLatentMixerConfig, HybridLatentTemporalMixer


class HybridLatentProjectionHead(nn.Module):
    """Project each mixed EEG feature into the BOLD latent space."""

    def __init__(
        self,
        hidden_dim: int = 256,
        latent_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or latent_dim <= 0:
            raise ValueError("hidden_dim and latent_dim must be positive integers")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output_projection(
            self.dropout(self.activation(self.hidden_projection(self.norm(value))))
        )


class HybridEEG2BOLDLatentModel(nn.Module):

    def __init__(
        self,
        eeg_tokenizer: Optional[nn.Module] = None,
        grounding: Optional[nn.Module] = None,
        latent_mixer: Optional[nn.Module] = None,
        bold_vae: Optional[nn.Module] = None,
        latent_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if eeg_tokenizer is None:
            raise ValueError(
                "Provide an EEG tokenizer"
            )
        if bold_vae is None:
            raise ValueError(
                "Provide a BOLD VAE, typically from load_frozen_bold_vae()"
            )
        self.eeg_tokenizer = eeg_tokenizer
        tokenizer_config = getattr(eeg_tokenizer, "config", None)
        hidden_dim = int(getattr(tokenizer_config, "hidden_dim", 256))
        slots = int(getattr(tokenizer_config, "slots_per_branch", 8)) * 2
        self.grounding = grounding or OverlappingLocalCrossAttention(
            HybridGroundingConfig(hidden_dim=hidden_dim, slots_per_frame=slots)
        )
        self.latent_mixer = latent_mixer or HybridLatentTemporalMixer(
            HybridLatentMixerConfig(hidden_dim=hidden_dim, ffn_dim=4 * hidden_dim, local_expansion_dim=2 * hidden_dim)
        )
        self.bold_vae = bold_vae
        self.latent_head = latent_head or HybridLatentProjectionHead(
            hidden_dim=hidden_dim,
            latent_dim=16,
        )
        vae_config = getattr(self.bold_vae, "config", None)
        if isinstance(vae_config, BOLDVAEConfig):
            _validate_bold_vae_config(vae_config)
        _freeze_eval(self.bold_vae)

    def train(self, mode: bool = True) -> "HybridEEG2BOLDLatentModel":
        """Keep the frozen BOLD VAE in evaluation mode."""
        super().train(mode)
        self.bold_vae.eval()
        return self

    @staticmethod
    def _check_eeg(eeg: torch.Tensor) -> None:
        expected = (512, 26, 16, 200)
        if not torch.is_tensor(eeg) or eeg.ndim != 5 or tuple(eeg.shape[1:]) != expected:
            raise ValueError(
                "Expected EEG shape [B, 512, 26, 16, 200], got {}".format(
                    tuple(eeg.shape) if torch.is_tensor(eeg) else type(eeg)
                )
            )

    @staticmethod
    def _check_bold(bold: torch.Tensor, batch_size: int) -> None:
        expected = (512, 64)
        if not torch.is_tensor(bold) or bold.ndim != 3 or tuple(bold.shape[1:]) != expected:
            raise ValueError(
                "Expected BOLD shape [B, 512, 64], got {}".format(
                    tuple(bold.shape) if torch.is_tensor(bold) else type(bold)
                )
            )
        if bold.shape[0] != batch_size:
            raise ValueError("EEG and BOLD must have the same batch size")

    def forward(
        self,
        eeg: torch.Tensor,
        bold: Optional[torch.Tensor] = None,
    ) -> EEG2BOLDOutput:
        """Predict BOLD and optionally encode its target latents."""
        self._check_eeg(eeg)
        if bold is not None:
            self._check_bold(bold, eeg.shape[0])

        eeg_tokens = self.eeg_tokenizer(eeg)
        grounded_hidden = self.grounding(eeg_tokens)
        mixed_hidden = self.latent_mixer(grounded_hidden)
        z_pred = self.latent_head(mixed_hidden)
        bold_pred = self.bold_vae.decode(z_pred)

        z_target = None
        if bold is not None:
            with torch.no_grad():
                z_target = self.bold_vae.encode(bold, sample=False)

        expected_z = (eeg.shape[0], 256, 16)
        expected_bold = (eeg.shape[0], 512, 64)
        if tuple(z_pred.shape) != expected_z:
            raise RuntimeError("Expected z_pred shape {}, got {}".format(expected_z, tuple(z_pred.shape)))
        if tuple(bold_pred.shape) != expected_bold:
            raise RuntimeError(
                "Expected bold_pred shape {}, got {}".format(
                    expected_bold,
                    tuple(bold_pred.shape),
                )
            )
        if z_target is not None and tuple(z_target.shape) != expected_z:
            raise RuntimeError(
                "Expected z_target shape {}, got {}".format(
                    expected_z,
                    tuple(z_target.shape),
                )
            )
        return EEG2BOLDOutput(
            z_pred=z_pred,
            bold_pred=bold_pred,
            z_target=z_target,
            shape_trace={
                "eeg_input": tuple(eeg.shape),
                "eeg_tokens": tuple(eeg_tokens.shape),
                "grounded_hidden": tuple(grounded_hidden.shape),
                "mixed_hidden": tuple(mixed_hidden.shape),
                "z_pred": tuple(z_pred.shape),
                "bold_pred": tuple(bold_pred.shape),
                "z_target": tuple(z_target.shape) if z_target is not None else None,
            },
        )


__all__ = [
    "EEG2BOLDOutput",
    "HybridEEG2BOLDLatentModel",
    "HybridLatentProjectionHead",
    "HybridEEGTokenizer",
    "HybridTokenizerConfig",
    "load_frozen_bold_vae",
]
