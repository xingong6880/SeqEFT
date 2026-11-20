"""在随机合成输入上检查真实模型的形状与梯度，不测量预测精度。

运行：python -m seqeft.smoke
不需要数据集或 checkpoint；完整 512 帧 EEG 编码较慢，因此只单独编码一帧，
随后在完整长度的随机 tokens 上检查 latent alignment 和冻结 decoder。
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.nn import functional as F

from .model import build_model
from .models.bold_vae import BOLDVAE, BOLDVAEConfig, functional_connectivity_mse_loss


def _finite_nonzero(value: torch.Tensor | None) -> bool:
    return value is not None and bool(torch.isfinite(value).all()) and bool(value.abs().sum() > 0)


def check_vae() -> dict:
    """Stage 1：在真实 512×64 输入上反传，检查 encoder 和 decoder。"""
    vae = BOLDVAE(BOLDVAEConfig(
        input_length=512, latent_length=256, temporal_compression=2,
    )).train()
    bold = torch.randn(1, 512, 64)
    output = vae(bold)
    assert output.reconstruction.shape == (1, 512, 64)
    assert output.mu.shape == output.z.shape == (1, 256, 16)
    assert torch.isfinite(output.total_loss)
    output.total_loss.backward()
    assert any(_finite_nonzero(p.grad) for p in vae.encoder.parameters())
    assert any(_finite_nonzero(p.grad) for p in vae.decoder.parameters())
    return {"bold_shape": [1, 512, 64], "latent_shape": [1, 256, 16],
            "encoder_gradient": True, "decoder_gradient": True}


def check_parameter_counts(model) -> dict:
    """参数量必须与论文对应的 Content Query + Local-only 实验一致。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert (total, trainable) == (18_147_696, 16_852_240), (total, trainable)
    assert not any(".global_attention." in name for name, _ in model.latent_mixer.named_parameters())
    assert torch.count_nonzero(model.grounding.content_projection.weight) == 0
    return {"total": total, "trainable": trainable}


def check_eeg_branches(model) -> dict:
    """一帧真实 LaBraM/MSS forward；使用随机权重和随机 EEG。"""
    model.eval()
    frame = torch.randn(1, 26, 16, 200)
    with torch.inference_mode():
        temporal = model.eeg_tokenizer.temporal.forward_patch_tokens(frame)
        spectral = model.eeg_tokenizer.spectral.forward_tokens(frame.reshape(1, 26, 3200))
        fused = model.eeg_tokenizer._forward_chunk(frame)
    assert temporal.shape == (1, 416, 200)
    assert spectral.shape == (1, 832, 200)
    assert fused.shape == (1, 16, 256)
    assert all(torch.isfinite(x).all() for x in (temporal, spectral, fused))
    return {"temporal_tokens": [1, 416, 200], "spectral_tokens": [1, 832, 200],
            "fused_tokens": [1, 16, 256]}


def check_alignment(model) -> dict:
    """检查完整长度对齐链路，以及 FC loss 穿过冻结 decoder 的梯度。"""
    model.zero_grad(set_to_none=True)
    model.train()
    assert not model.bold_vae.training
    assert all(not p.requires_grad for p in model.bold_vae.parameters())
    tokens = torch.randn(1, 512, 16, 256, requires_grad=True)
    target = torch.randn(1, 512, 64)
    hidden = model.grounding(tokens)
    mixed = model.latent_mixer(hidden)
    latent = model.latent_head(mixed)
    prediction = model.bold_vae.decode(latent)
    with torch.no_grad():
        latent_target = model.bold_vae.encode(target, sample=False)
    assert hidden.shape == mixed.shape == (1, 256, 256)
    assert latent.shape == latent_target.shape == (1, 256, 16)
    assert prediction.shape == target.shape == (1, 512, 64)
    assert not latent_target.requires_grad
    fc_loss = functional_connectivity_mse_loss(prediction, target)
    fc_gradient = torch.autograd.grad(fc_loss, latent, retain_graph=True)[0]
    assert _finite_nonzero(fc_gradient), "FC loss 必须穿过冻结 decoder 传到 latent"
    loss = F.mse_loss(latent, latent_target) + 0.1 * fc_loss
    assert torch.isfinite(loss)
    loss.backward()
    assert _finite_nonzero(model.latent_head.output_projection.weight.grad)
    assert _finite_nonzero(model.grounding.content_projection.weight.grad)
    assert _finite_nonzero(model.grounding.cross_attention.in_proj_weight.grad)
    assert _finite_nonzero(model.latent_mixer.blocks[0].depthwise_conv.weight.grad)
    assert _finite_nonzero(tokens.grad)
    assert all(p.grad is None for p in model.bold_vae.parameters())
    model.zero_grad(set_to_none=True)
    return {"input_tokens": [1, 512, 16, 256], "latent_shape": [1, 256, 16],
            "bold_shape": [1, 512, 64], "fc_gradient_through_decoder": True,
            "grounding_gradient": True, "mixer_gradient": True,
            "projection_gradient": True, "vae_frozen_and_eval": True}


def run_smoke(*, include_eeg: bool = True) -> dict:
    """执行 CPU 合成检查；真实数据与预训练性能仍需独立评价。"""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    try:
        torch.manual_seed(2026)
        result = {"inputs": "synthetic random data", "weights": "random initialization",
                  "device": "cpu", "stage1": check_vae()}
        model = build_model({}, load_pretrained=False)
        result["parameters"] = check_parameter_counts(model)
        if include_eeg:
            result["single_frame_eeg"] = check_eeg_branches(model)
        result["stage2_alignment"] = check_alignment(model)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["status"] = "passed"
        return result
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-eeg", action="store_true",
                        help="跳过单帧 LaBraM/MSS 检查，只验证 VAE 与 latent alignment")
    args = parser.parse_args()
    print(json.dumps(run_smoke(include_eeg=not args.skip_eeg), indent=2))


if __name__ == "__main__":
    main()
