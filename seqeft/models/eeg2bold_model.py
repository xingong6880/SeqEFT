# SPDX-License-Identifier: GPL-3.0-only
"""Frozen BOLD VAE loading and model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from .bold_vae import BOLDVAE, BOLDVAEConfig



def _validate_bold_vae_config(config: BOLDVAEConfig) -> None:
    expected = {
        "input_length": 512,
        "latent_length": 256,
        "temporal_compression": 2,
        "roi_count": 64,
        "latent_dim": 16,
    }
    for name, value in expected.items():
        if getattr(config, name, None) != value:
            raise ValueError(
                "The reference BOLD VAE requires {}={}, got {}".format(
                    name,
                    value,
                    getattr(config, name, None),
                )
            )


_CHECKPOINT_ARCHITECTURE_FIELDS = (
    "input_length",
    "latent_length",
    "temporal_compression",
    "roi_count",
    "latent_dim",
    "hidden_width",
    "attention_heads",
    "bottleneck_depth",
    "ffn_dim",
    "dropout",
    "resblock_depth",
)


def _validate_checkpoint_metadata(
    payload: object,
    config: BOLDVAEConfig,
) -> None:
    """Check architecture metadata that tensor shapes alone cannot validate."""
    if not isinstance(payload, Mapping):
        return

    metadata_sources = []
    metadata = payload.get("config")
    if isinstance(metadata, Mapping):
        model_metadata = metadata.get("model")
        if isinstance(model_metadata, Mapping):
            metadata_sources.append(model_metadata)
        metadata_sources.append(metadata)
    metadata_sources.append(payload)

    for source in metadata_sources:
        for name in _CHECKPOINT_ARCHITECTURE_FIELDS:
            if name not in source:
                continue
            checkpoint_value = source[name]
            requested_value = getattr(config, name)
            if checkpoint_value != requested_value:
                raise ValueError(
                    "checkpoint BOLD-VAE config mismatch for {}: "
                    "checkpoint={} requested={}".format(
                        name,
                        checkpoint_value,
                        requested_value,
                    )
                )


def _freeze_eval(module: nn.Module) -> None:
    """Freeze module parameters and enter evaluation mode."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def load_frozen_bold_vae(
    checkpoint_path: Path,
    config: Optional[BOLDVAEConfig] = None,
    *,
    device: Optional[torch.device] = None,
) -> BOLDVAE:
    """Load and validate a compatible state dictionary for the frozen BOLD VAE."""
    if config is None:
        config = BOLDVAEConfig(
            input_length=512,
            latent_length=256,
            temporal_compression=2,
        )
    _validate_bold_vae_config(config)
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError("BOLD VAE checkpoint does not exist: {}".format(path))

    model = BOLDVAE(config)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_metadata(payload, config)
    if isinstance(payload, Mapping) and "model" in payload:
        state = payload["model"]
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, Mapping):
        raise RuntimeError("The BOLD VAE checkpoint has no usable state dictionary")
    model.load_state_dict(state, strict=True)
    _freeze_eval(model)
    if device is not None:
        model.to(device)
        model.eval()
    return model


@dataclass
class EEG2BOLDOutput:
    """Predicted latents and BOLD, with optional target latents and shape trace."""

    z_pred: torch.Tensor
    bold_pred: torch.Tensor
    z_target: Optional[torch.Tensor] = None
    shape_trace: Optional[dict] = None

    def __getitem__(self, key: str) -> Any:
        if key not in {"z_pred", "bold_pred", "z_target", "shape_trace"}:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
