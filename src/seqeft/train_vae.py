"""Stage 1：只用训练扫描预训练 fMRI 序列 VAE。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from .data import RandomFixedWindowBOLDDataset, load_existing_bold_caches
from .models.bold_vae import BOLDVAE, BOLDVAEConfig
from .utils import configure_runtime, load_config, restore_rng, rng_state, save_checkpoint, write_json


def run(config, device="cuda", resume=None):
    seed, training = config["experiment"]["seed"], config["training"]
    if training.get("gradient_accumulation_steps", 1) != 1 or training.get("num_workers", 0) != 0:
        raise ValueError("参考协议使用 accumulation=1、num_workers=0")
    configure_runtime(seed)
    device = torch.device(device)
    caches = load_existing_bold_caches(config, split="train")
    dataset = RandomFixedWindowBOLDDataset(caches, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True,
                            num_samples=training["samples_per_epoch"], generator=generator)
    loader = DataLoader(dataset, sampler=sampler, batch_size=training["batch_size"],
                        num_workers=0, generator=generator)
    model = BOLDVAE(BOLDVAEConfig(**config["model"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    output_dir = Path(config["experiment"]["output_root"])
    start, step, history = 0, 0, []
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload["config"] != config:
            raise ValueError("恢复训练时配置必须与 checkpoint 完全一致")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"], generator)
        start, step, history = payload["epoch"], payload["step"], payload["history"]
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", config)
    for epoch in range(start + 1, training["epochs"] + 1):
        # Stage 1 原始 crop 序列从 epoch=1 开始；Stage 2 从 epoch=0 开始。
        dataset.set_epoch(epoch)
        model.train()
        sums = torch.zeros(4, device=device)
        optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            result = model(batch["bold"].to(device))
            if not torch.isfinite(result.total_loss):
                raise ValueError("VAE loss 出现 NaN/Inf")
            result.total_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            sums += torch.stack([x.detach() for x in
                                 (result.total_loss, result.recon_loss, result.spectral_loss, result.kl_loss)])
        values = (sums / len(loader)).cpu().tolist()
        row = dict(zip(("loss", "reconstruction_mse", "spectral_loss", "kl"), values))
        row.update(epoch=epoch, global_step=step)
        history.append(row)
        save_checkpoint(output_dir / "last.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
            "epoch": epoch, "step": step, "history": history, "rng": rng_state(generator),
            "window_length": 512, "latent_length": 256, "temporal_compression": 2,
            "latent_dim": 16, "checkpoint_kind": "vae_training",
        })
        write_json(output_dir / "history.json", history)
        print(row, flush=True)
    return {"checkpoint": str(output_dir / "last.pt"), "updates": step}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vae.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="同一配置的 last.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config["experiment"]["output_root"] = args.output_dir
    print(run(config, args.device, args.resume))


if __name__ == "__main__":
    main()
