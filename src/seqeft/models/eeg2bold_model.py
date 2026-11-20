# SPDX-License-Identifier: GPL-3.0-only
# 从论文研究实现移入；仅调整包内导入和公开说明，保留计算与参数名称。
"""冻结 fMRI VAE 的加载、检查与 SeqEFT 输出结构。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from .bold_vae import BOLDVAE, BOLDVAEConfig



def _validate_bold_vae_config(config: BOLDVAEConfig) -> None:
    """检查 EEG2BOLD V0 使用的 fixed V0-B latent contract。"""
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
                "BOLD-VAE V0-B contract 要求 {}={}，实际为 {}".format(
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
    """校验 checkpoint 保存的 architecture metadata，避免同 shape 错配。"""
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
    """冻结参数，并保证 dropout/normalization 不随父模型切回 train。"""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def load_frozen_bold_vae(
    checkpoint_path: Path,
    config: Optional[BOLDVAEConfig] = None,
    *,
    device: Optional[torch.device] = None,
) -> BOLDVAE:
    """严格加载 V0-B BOLD-VAE checkpoint；不存在或不匹配时直接失败。"""
    if config is None:
        config = BOLDVAEConfig(
            input_length=512,
            latent_length=256,
            temporal_compression=2,
        )
    _validate_bold_vae_config(config)
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError("BOLD-VAE checkpoint 不存在: {}".format(path))

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
        raise RuntimeError("BOLD-VAE checkpoint 不包含可用的 state dict")
    model.load_state_dict(state, strict=True)
    _freeze_eval(model)
    if device is not None:
        model.to(device)
        model.eval()
    return model


@dataclass
class EEG2BOLDOutput:
    """总模型输出；同时支持属性访问和简单的 key 访问。"""

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
