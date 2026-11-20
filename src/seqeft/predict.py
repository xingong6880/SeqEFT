"""使用完整 EMA checkpoint，从已对齐的 EEG 数组预测 512 帧、64 ROI BOLD。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .utils import load_inference_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eeg", required=True, help="float32 .npy，形状 [512,26,16,200]，单位 µV")
    parser.add_argument("--output", required=True, help="输出 .npy，形状 [512,64]")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    eeg = np.load(args.eeg, allow_pickle=False)
    if eeg.shape != (512, 26, 16, 200) or not np.isfinite(eeg).all():
        raise ValueError("EEG 必须为有限的 [512,26,16,200] 数组；预处理见 docs/data.md")
    output_path = Path(args.output)
    if output_path.suffix != ".npy":
        raise ValueError("--output 必须以 .npy 结尾")
    if output_path.exists():
        raise FileExistsError(output_path)
    model, _ = load_inference_model(args.checkpoint, args.device)
    with torch.no_grad():
        prediction = model(torch.as_tensor(eeg, dtype=torch.float32, device=args.device)[None]).bold_pred[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, prediction.cpu().numpy())
    print(f"Saved {tuple(prediction.shape)} normalized BOLD predictions to {output_path}")


if __name__ == "__main__":
    main()
