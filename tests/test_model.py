"""Check reference model shapes and gradients with synthetic inputs."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from seqeft.model import build_model
from seqeft.models.bold_vae import BOLDVAE, functional_connectivity_mse_loss


def _finite_nonzero(value: torch.Tensor | None) -> bool:
    return value is not None and bool(torch.isfinite(value).all()) and bool(value.abs().sum() > 0)


@pytest.fixture(scope="module")
def reference_model():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield build_model({}, load_pretrained=False)
    finally:
        torch.set_num_threads(previous_threads)


def test_stage1_forward_backward(reference_model):
    vae = BOLDVAE(reference_model.bold_vae.config).train()
    output = vae(torch.randn(1, 512, 64))
    assert output.reconstruction.shape == (1, 512, 64)
    assert output.mu.shape == output.z.shape == (1, 256, 16)
    assert torch.isfinite(output.total_loss)
    output.total_loss.backward()
    assert any(_finite_nonzero(p.grad) for p in vae.encoder.parameters())
    assert any(_finite_nonzero(p.grad) for p in vae.decoder.parameters())


def test_reference_architecture_and_initialization(reference_model):
    total = sum(p.numel() for p in reference_model.parameters())
    trainable = sum(p.numel() for p in reference_model.parameters() if p.requires_grad)
    assert (total, trainable) == (18_147_696, 16_852_240)
    assert not any(
        ".global_attention." in name for name, _ in reference_model.latent_mixer.named_parameters()
    )
    assert torch.count_nonzero(reference_model.grounding.content_projection.weight) == 0


def test_real_eeg_branches_single_frame(reference_model):
    tokenizer = reference_model.eval().eeg_tokenizer
    frame = torch.randn(1, 26, 16, 200)
    with torch.inference_mode():
        temporal = tokenizer.temporal.forward_patch_tokens(frame)
        spectral = tokenizer.spectral.forward_tokens(frame.reshape(1, 26, 3200))
        fused = tokenizer._forward_chunk(frame)
    assert temporal.shape == (1, 416, 200)
    assert spectral.shape == (1, 832, 200)
    assert fused.shape == (1, 16, 256)
    assert all(torch.isfinite(x).all() for x in (temporal, spectral, fused))


def test_alignment_backpropagates_through_frozen_decoder(reference_model):
    model = reference_model.train()
    model.zero_grad(set_to_none=True)
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
    assert _finite_nonzero(fc_gradient), "The FC loss must propagate through the frozen decoder to the latent."
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
