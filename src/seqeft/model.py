"""构造论文使用的 SeqEFT；保留历史 checkpoint 的参数名称与初始化顺序。"""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from .models.bold_vae import BOLDVAE, BOLDVAEConfig
from .models.content_query_grounding import ContentQueryGrounding
from .models.eeg2bold_model import load_frozen_bold_vae
from .models.hybrid_eeg2bold_model import HybridEEG2BOLDLatentModel
from .models.hybrid_eeg_tokenizer import HybridEEGTokenizer, HybridTokenizerConfig
from .models.hybrid_grounding import HybridGroundingConfig, OverlappingLocalCrossAttention
from .models.hybrid_latent_head import HybridLatentProjectionHead
from .models.hybrid_latent_mixer import HybridLatentMixerConfig, HybridLatentTemporalMixer
from .models.labram import LaBraMTemporalEncoder
from .models.local_only_latent_mixer import LocalOnlyLatentTemporalMixer
from .models.spectral import MultiScaleSpectralEncoder


SeqEFT = HybridEEG2BOLDLatentModel


def seed_everything(seed: int) -> None:
    """同步 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_model(
    config: dict,
    device: torch.device | str = "cpu",
    *,
    load_pretrained: bool = True,
) -> SeqEFT:
    """构造 Content Query + Local-only 模型，并冻结 fMRI VAE。

    默认加载 LaBraM 与 Stage 1 VAE 权重。只有验证或即将严格加载完整
    SeqEFT checkpoint 时才使用 load_pretrained=False；该模式不提供训练权重。
    config 使用公开 YAML 的 experiment 和 model 两个字典。
    """
    settings = config.get("model", {})
    seed_everything(config.get("experiment", {}).get("seed", 20261012))
    width = settings.get("hidden_dim", 256)
    heads = settings.get("heads", 8)
    slots = settings.get("slots_per_branch", 8)
    dropout = settings.get("dropout", 0.1)
    tokenizer_config = HybridTokenizerConfig(
        hidden_dim=width,
        input_scale=settings.get("input_scale", 0.01),
        frame_chunk_size=settings.get("frame_chunk_size", 16),
        slots_per_branch=slots,
        depth=settings.get("tokenizer_depth", 2),
        heads=heads,
        dropout=dropout,
        samples_per_patch=settings.get("feature_dim", 200),
        temporal_token_count=settings.get("temporal_token_count", 416),
        spectral_token_count=settings.get("spectral_token_count", 832),
    )
    freeze_temporal = not settings.get("train_labram", True)
    if load_pretrained:
        tokenizer = HybridEEGTokenizer.from_checkpoint(
            Path(settings["labram_checkpoint"]),
            config=tokenizer_config,
            freeze_temporal=freeze_temporal,
        )
        bold_vae = load_frozen_bold_vae(Path(settings["bold_vae_checkpoint"]))
    else:
        # 与 from_checkpoint 保持同样的构造顺序，MSS 独立使用原始固定种子。
        temporal = LaBraMTemporalEncoder()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            spectral = MultiScaleSpectralEncoder()
        tokenizer = HybridEEGTokenizer(
            temporal=temporal,
            spectral=spectral,
            config=tokenizer_config,
            freeze_temporal=freeze_temporal,
        )
        bold_vae = BOLDVAE(
            BOLDVAEConfig(input_length=512, latent_length=256, temporal_compression=2)
        )

    # 先按论文实验初始化公共模型，再移除 global mixer attention。
    # 直接初始化简化网络会改变 RNG 消耗，从而改变随后模块的初值。
    grounding = OverlappingLocalCrossAttention(
        HybridGroundingConfig(
            hidden_dim=width,
            heads=heads,
            slots_per_frame=2 * slots,
            context_radius=settings.get("context_radius", 2),
            dropout=dropout,
        )
    )
    mixer = HybridLatentTemporalMixer(
        HybridLatentMixerConfig(
            hidden_dim=width,
            depth=settings.get("mixer_depth", 4),
            heads=heads,
            ffn_dim=settings.get("mixer_ffn_dim", 1024),
            local_expansion_dim=settings.get("local_expansion_dim", 512),
            kernel_size=settings.get("local_kernel_size", 5),
            dropout=dropout,
        )
    )
    head = HybridLatentProjectionHead(
        hidden_dim=width,
        latent_dim=settings.get("latent_dim", 16),
        dropout=dropout,
    )
    model = HybridEEG2BOLDLatentModel(
        eeg_tokenizer=tokenizer,
        grounding=grounding,
        latent_mixer=mixer,
        bold_vae=bold_vae,
        latent_head=head,
    ).to(torch.device(device))
    model.grounding = ContentQueryGrounding.from_position_grounding(
        model.grounding
    ).to(device=next(model.grounding.parameters()).device,
         dtype=next(model.grounding.parameters()).dtype)
    model.latent_mixer, _ = LocalOnlyLatentTemporalMixer.from_hybrid(model.latent_mixer)
    model.bold_vae.eval()
    return model


__all__ = ["SeqEFT", "build_model", "seed_everything"]
