"""Predict regional BOLD from a preprocessed EEG window using an EMA checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .utils import load_inference_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eeg", required=True, help="EEG .npy array in microvolts with shape [512, 26, 16, 200]")
    parser.add_argument("--output", required=True, help="Output .npy array with shape [512, 64]")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    eeg = np.load(args.eeg, allow_pickle=False)
    if eeg.shape != (512, 26, 16, 200) or not np.isfinite(eeg).all():
        raise ValueError("EEG must be a finite [512, 26, 16, 200] array in microvolts. See README.md.")
    output_path = Path(args.output)
    if output_path.suffix != ".npy":
        raise ValueError("--output must end in .npy")
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
