"""配置、随机状态与 checkpoint 的小型公共工具。"""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml


def load_config(path):
    """路径相对于运行命令的目录；README 命令均在仓库根目录执行。"""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("配置文件必须是 YAML 字典")
    return config


def configure_runtime(seed):
    """与论文运行一致：FP32、单 CPU 线程及 high matmul precision。"""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def save_checkpoint(path, payload):
    """先写临时文件再替换，避免中断产生半个 checkpoint。"""
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
    """完整 SeqEFT checkpoint 内含 VAE；推理无需另取 LaBraM/VAE 文件。"""
    from .model import build_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configure_runtime(payload["config"].get("experiment", {}).get("seed", 20261012))
    model = build_model(payload["config"], device=device, load_pretrained=False)
    weights = dict(payload["model"])
    # last.pt 保存训练参数及 EMA；推理始终使用 EMA shadow 覆盖可训练参数。
    if "ema" in payload:
        if payload["ema"]["num_updates"] <= 0:
            raise ValueError("checkpoint 中的 EMA 尚未更新")
        weights.update(payload["ema"]["shadow"])
    elif payload.get("checkpoint_kind") != "ema_inference_only":
        raise ValueError("请提供 ema_last.pt 或包含 EMA 状态的训练 checkpoint")
    model.load_state_dict(weights, strict=True)
    return model.eval(), payload["config"]
