"""Configuration, checkpoint, and RNG utilities."""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml


def load_config(path):
    """Load a YAML config with paths relative to the working directory."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The config must be a YAML mapping")
    return config


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(seed):
    """Set the reference precision, thread count, and random seeds."""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(seed)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def save_checkpoint(path, payload):
    """Replace the checkpoint atomically after writing the complete file."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "generator": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["generator"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_inference_model(checkpoint, device):
    """Load a complete SeqEFT model with EMA weights, including the frozen VAE."""
    from .model import build_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configure_runtime(payload["config"].get("experiment", {}).get("seed", 20261012))
    model = build_model(payload["config"], device=device, load_pretrained=False)
    weights = dict(payload["model"])
    # Overlay the EMA weights on a training checkpoint.
    if "ema" in payload:
        if payload["ema"]["num_updates"] <= 0:
            raise ValueError("The checkpoint EMA has received no updates")
        weights.update(payload["ema"]["shadow"])
    elif payload.get("checkpoint_kind") != "ema_inference_only":
        raise ValueError("Provide ema_last.pt or a training checkpoint that contains EMA state")
    model.load_state_dict(weights, strict=True)
    return model.eval(), payload["config"]
