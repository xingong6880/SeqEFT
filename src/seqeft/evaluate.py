"""按扫描计算 Center512 六项指标；多个 checkpoint 汇总 seed 均值与样本 SD。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CenterPairedWindowDataset, load_existing_paired_caches, paired_collate
from .metrics import scan_metrics
from .utils import load_config, load_inference_model, write_json


@torch.no_grad()
def evaluate(checkpoint, config, device="cuda", split="test"):
    model, _ = load_inference_model(checkpoint, device)
    dataset = CenterPairedWindowDataset(load_existing_paired_caches(config, split=split))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=paired_collate)
    rows = []
    for batch in loader:
        prediction = model(batch["eeg"].to(device)).bold_pred[0].cpu().numpy().astype(np.float64)
        target = batch["bold"][0].numpy().astype(np.float64)
        scores = scan_metrics(prediction, target)
        rows.append({"scan_id": batch["scan_id"][0], "subject_id": batch["subject_id"][0], **scores})
    keys = ("tcorr", "bold_mse", "pixcorr", "conn_mse", "f1_025", "f1_05")
    return {"checkpoint": str(checkpoint), "split": split, "scans": rows,
            "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seqeft.yaml")
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output", default="evaluation.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.config)
    runs = [evaluate(path, config, args.device, args.split) for path in args.checkpoint]
    result = {"runs": runs, "aggregation": "scan mean within seed; sample SD across seeds",
              "mean": {k: float(np.mean([r["mean"][k] for r in runs])) for k in runs[0]["mean"]}}
    if len(runs) > 1:
        result["sample_sd"] = {k: float(np.std([r["mean"][k] for r in runs], ddof=1))
                               for k in runs[0]["mean"]}
    write_json(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
