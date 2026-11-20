# SPDX-License-Identifier: GPL-3.0-only
# 从论文研究实现移入；仅调整包内导入和公开说明，保留计算与参数名称。
"""Hybrid V1 EEG -> BOLD latent 总模型。"""

from __future__ import annotations

from typing import Any, Optional

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
from .hybrid_latent_head import HybridLatentProjectionHead
from .hybrid_latent_mixer import HybridLatentMixerConfig, HybridLatentTemporalMixer


class HybridEEG2BOLDLatentModel(nn.Module):
    """V1-Base 主链路；不修改 V0 的 model/training build path。"""

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
                "必须显式提供已按 checkpoint 构造的 eeg_tokenizer；测试可注入 fake tokenizer"
            )
        if bold_vae is None:
            raise ValueError(
                "必须显式提供已 strict-load 的 V0-B bold_vae；请先调用 load_frozen_bold_vae"
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
        """父模型切换 train 时，BOLD-VAE 仍保持 frozen eval。"""
        super().train(mode)
        self.bold_vae.eval()
        return self

    @staticmethod
    def _check_eeg(eeg: torch.Tensor) -> None:
        expected = (512, 26, 16, 200)
        if not torch.is_tensor(eeg) or eeg.ndim != 5 or tuple(eeg.shape[1:]) != expected:
            raise ValueError(
                "EEG 输入必须是 [B,512,26,16,200]，实际为 {}".format(
                    tuple(eeg.shape) if torch.is_tensor(eeg) else type(eeg)
                )
            )

    @staticmethod
    def _check_bold(bold: torch.Tensor, batch_size: int) -> None:
        expected = (512, 64)
        if not torch.is_tensor(bold) or bold.ndim != 3 or tuple(bold.shape[1:]) != expected:
            raise ValueError(
                "BOLD 输入必须是 [B,512,64]，实际为 {}".format(
                    tuple(bold.shape) if torch.is_tensor(bold) else type(bold)
                )
            )
        if bold.shape[0] != batch_size:
            raise ValueError("EEG 和 BOLD 的 batch size 必须一致")

    def forward(
        self,
        eeg: torch.Tensor,
        bold: Optional[torch.Tensor] = None,
    ) -> EEG2BOLDOutput:
        """执行 V1 forward；loss 由 runner 在模型外计算。"""
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
            raise RuntimeError("z_pred shape 必须是 {}，实际为 {}".format(expected_z, tuple(z_pred.shape)))
        if tuple(bold_pred.shape) != expected_bold:
            raise RuntimeError(
                "bold_pred shape 必须是 {}，实际为 {}".format(
                    expected_bold,
                    tuple(bold_pred.shape),
                )
            )
        if z_target is not None and tuple(z_target.shape) != expected_z:
            raise RuntimeError(
                "z_target shape 必须是 {}，实际为 {}".format(
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
    "HybridEEGTokenizer",
    "HybridTokenizerConfig",
    "load_frozen_bold_vae",
]
