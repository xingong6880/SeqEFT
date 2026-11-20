"""Stage 2：训练 EEG 到 fMRI latent 的映射，冻结 VAE 并维护 EMA。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import weakref

import torch
from torch.utils.data import DataLoader

from .data import RandomPairedWindowDataset, load_existing_paired_caches, paired_collate
from .model import build_model
from .models.bold_vae import functional_connectivity_mse_loss
from .utils import configure_runtime, load_config, restore_rng, rng_state, save_checkpoint, write_json


class TrainableEMA:
    """论文 EMA：第一次更新后复制参数，此后以 0.99 平滑；VAE 和 buffers 不参与。"""

    def __init__(self, model, decay=0.99):
        self.model_ref = weakref.ref(model)
        self.decay = decay
        self.num_updates = 0
        self.shadow = {}

    @torch.no_grad()
    def update(self):
        for name, parameter in self.model_ref().named_parameters():
            if parameter.requires_grad:
                if self.num_updates == 0:
                    self.shadow[name] = parameter.detach().clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(parameter, alpha=1-self.decay)
        self.num_updates += 1

    def state_dict(self):
        return {"decay": self.decay, "num_updates": self.num_updates,
                "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state):
        parameters = dict(self.model_ref().named_parameters())
        expected = {k for k, p in parameters.items() if p.requires_grad}
        if state["decay"] != self.decay or set(state["shadow"]) != expected:
            raise ValueError("EMA 参数集合或 decay 不匹配")
        self.num_updates = state["num_updates"]
        self.shadow = {k: v.to(parameters[k]).clone() for k, v in state["shadow"].items()}

    @contextmanager
    def applied(self):
        if not self.num_updates:
            raise RuntimeError("EMA 尚未更新")
        parameters = dict(self.model_ref().named_parameters())
        original = {k: parameters[k].detach().clone() for k in self.shadow}
        try:
            with torch.no_grad():
                for key, value in self.shadow.items():
                    parameters[key].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for key, value in original.items():
                    parameters[key].copy_(value)


def build_optimizer(model, config):
    """保持论文三组参数的顺序、学习率与 weight decay。"""
    training = config["training"]
    groups = [
        {"name": "pretrained_labram",
         "params": [p for p in model.eeg_tokenizer.temporal.parameters() if p.requires_grad],
         "lr": training["pretrained_labram_learning_rate"]},
        {"name": "eeg_task_specific",
         "params": [p for n, p in model.eeg_tokenizer.named_parameters()
                    if not n.startswith("temporal.") and p.requires_grad],
         "lr": training["eeg_learning_rate"]},
        {"name": "translator",
         "params": [p for part in (model.grounding, model.latent_mixer, model.latent_head)
                    for p in part.parameters() if p.requires_grad],
         "lr": training["main_learning_rate"]},
    ]
    ids = [id(p) for group in groups for p in group["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("optimizer 必须恰好覆盖所有可训练参数")
    return torch.optim.AdamW(groups, weight_decay=training["weight_decay"])


def build_scheduler(optimizer, config):
    warmup, total = config["training"]["warmup_epochs"], config["training"]["epochs"]

    def schedule(epoch):
        if epoch < warmup:
            return float(epoch + 1) / max(warmup, 1)
        progress = min(max((epoch - warmup) / max(total - warmup - 1, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def alignment_loss(output, bold, config):
    """decoder 参数被冻结，但保留对预测 latent 的梯度。"""
    latent = (output.z_pred.float() - output.z_target.float()).square().mean()
    fc = functional_connectivity_mse_loss(output.bold_pred, bold)
    loss = config["loss"]["latent_mse_weight"] * latent + config["loss"]["fc_weight"] * fc
    return loss, latent, fc


def run(config, device="cuda", resume=None):
    seed, training = config["experiment"]["seed"], config["training"]
    if training.get("gradient_accumulation_steps", 1) != 1 or training.get("num_workers", 0) != 0:
        raise ValueError("参考协议使用 accumulation=1、num_workers=0")
    configure_runtime(seed)
    device = torch.device(device)
    caches = load_existing_paired_caches(config, split="train")
    dataset = RandomPairedWindowDataset(caches, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=True,
                        drop_last=False, num_workers=0, generator=generator,
                        collate_fn=paired_collate)
    model = build_model(config, device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    ema = TrainableEMA(model, training.get("ema_decay", 0.99))
    output_dir = Path(config["experiment"]["output_root"])
    start, step, history = 0, 0, []
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload["config"] != config:
            raise ValueError("恢复训练时配置必须与 checkpoint 完全一致")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        ema.load_state_dict(payload["ema"])
        restore_rng(payload["rng"], generator)
        start, step, history = payload["epoch"], payload["global_step"], payload["history"]
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", config)
    for epoch in range(start, training["epochs"]):
        dataset.set_epoch(epoch)
        model.train()
        sums = torch.zeros(3, device=device)
        optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            eeg, bold = batch["eeg"].to(device), batch["bold"].to(device)
            loss, latent, fc = alignment_loss(model(eeg, bold), bold, config)
            if not torch.isfinite(loss):
                raise ValueError("训练 loss 出现 NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           training["grad_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            ema.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            sums += torch.stack((loss.detach(), latent.detach(), fc.detach()))
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        scheduler.step()
        values = (sums / len(loader)).cpu().tolist()
        row = {"epoch": epoch + 1, "global_step": step, "loss": values[0],
               "latent_mse": values[1], "conn_mse": values[2], "learning_rates": learning_rates}
        history.append(row)
        save_checkpoint(output_dir / "last.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "ema": ema.state_dict(),
            "config": config, "epoch": epoch + 1, "global_step": step,
            "history": history, "rng": rng_state(generator), "checkpoint_kind": "training",
        })
        write_json(output_dir / "history.json", history)
        print(row, flush=True)
    with ema.applied():
        save_checkpoint(output_dir / "ema_last.pt", {
            "model": model.state_dict(), "config": config, "epoch": training["epochs"],
            "global_step": step, "checkpoint_kind": "ema_inference_only",
        })
    return {"checkpoint": str(output_dir / "ema_last.pt"), "updates": step}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seqeft.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="同一配置的 last.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.output_dir is not None:
        config["experiment"]["output_root"] = args.output_dir
    print(run(config, args.device, args.resume))


if __name__ == "__main__":
    main()
