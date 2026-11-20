"""Test EMA, the reference learning rate schedule, and RNG restoration."""
import random

import numpy as np
import pytest
import torch
from torch import nn

from seqeft import train
from seqeft.model import build_model
from seqeft.train import TrainableEMA, build_scheduler
from seqeft.utils import restore_rng, rng_state


def test_ema_starts_after_first_update_and_restores():
    model = nn.Linear(1, 1, bias=False)
    model.register_parameter("frozen", nn.Parameter(torch.ones(1), requires_grad=False))
    ema = TrainableEMA(model, decay=0.99)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update()
    assert set(ema.shadow) == {"weight"}
    assert ema.shadow["weight"].item() == 2.0
    with torch.no_grad():
        model.weight.fill_(4.0)
    ema.update()
    assert ema.shadow["weight"].item() == pytest.approx(2.02)
    with pytest.raises(RuntimeError):
        with ema.applied():
            assert model.weight.item() == pytest.approx(2.02)
            raise RuntimeError("Simulated evaluation failure")
    assert model.weight.item() == 4.0
    assert ema.num_updates == 2


def test_reference_learning_rate_endpoints():
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_scheduler(optimizer, {"training": {"warmup_epochs": 5, "epochs": 150}})
    used = []
    for _ in range(150):
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert used[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert used[5] == 1.0
    assert used[-1] == pytest.approx(0.0)
    assert all(a >= b for a, b in zip(used[5:-1], used[6:]))


def test_rng_round_trip():
    generator = torch.Generator().manual_seed(42)
    state = rng_state(generator)
    expected = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator))
    restore_rng(state, generator)
    actual = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)


def test_stage_two_resume_uses_weights_from_training_checkpoint(tmp_path, monkeypatch):
    config = {
        "experiment": {"seed": 42, "output_root": str(tmp_path)},
        "model": {
            "labram_checkpoint": str(tmp_path / "missing_labram.pth"),
            "bold_vae_checkpoint": str(tmp_path / "missing_vae.pt"),
        },
        "training": {
            "batch_size": 1, "epochs": 1, "warmup_epochs": 0, "weight_decay": 0.01,
            "pretrained_labram_learning_rate": 1e-5, "eeg_learning_rate": 1e-4,
            "main_learning_rate": 3e-4,
        },
    }
    monkeypatch.setattr(train, "load_existing_paired_caches", lambda *_args, **_kwargs: [None])
    monkeypatch.setattr(train, "RandomPairedWindowDataset", lambda caches, *, seed: caches)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = build_model(config, load_pretrained=False)
        with torch.no_grad():
            model.latent_head.output_projection.bias.add_(0.125)
        optimizer = train.build_optimizer(model, config)
        ema = TrainableEMA(model)
        ema.update()
        checkpoint = tmp_path / "last.pt"
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": build_scheduler(optimizer, config).state_dict(),
            "ema": ema.state_dict(), "config": config, "epoch": 1, "global_step": 1,
            "history": [], "rng": rng_state(torch.Generator().manual_seed(42)),
            "checkpoint_kind": "training",
        }, checkpoint)
        summary = train.run(config, "cpu", resume=checkpoint)
        exported = torch.load(summary["checkpoint"], map_location="cpu", weights_only=False)
        assert summary["updates"] == 1
        assert exported["checkpoint_kind"] == "ema_inference_only"
        torch.testing.assert_close(exported["model"], model.state_dict(), rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
