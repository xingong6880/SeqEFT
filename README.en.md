# SeqEFT

**Latent Sequence Alignment for EEG-to-fMRI Translation**

[中文](README.md) · [Project page](https://xingong6880.github.io/projects/seqeft/) · [Data](docs/data.md) · [Reproducibility](docs/reproducibility.md)

SeqEFT reconstructs regional fMRI (BOLD) time series from EEG through latent sequence alignment.

## Method

1. **Pretrain an fMRI sequence VAE.** Encode a 512-frame, 64-ROI sequence into a 256-step, 16-dimensional latent sequence.
2. **Align EEG with fMRI latents.** A LaBraM branch and a multi-scale spectral branch encode EEG. Content-dependent local aggregation and temporal mixing predict the latent sequence, which a frozen fMRI decoder reconstructs into BOLD.

The alignment objective is latent MSE + 0.1 × ConnMSE. Inference uses EEG only. Each of the 512 target frames is paired with the preceding 16 seconds of 26-channel EEG.

## Install and smoke test

Reference environment: Python 3.9 and PyTorch 2.0.0. Run commands from the repository root.

```bash
git clone https://github.com/xingong6880/SeqEFT.git
cd SeqEFT
conda create -n seqeft python=3.9 -y
conda activate seqeft
pip install -e ".[test]"
python -m seqeft.smoke
python -m pytest -q
```

The smoke test uses synthetic data and random weights; it requires no dataset or pretrained checkpoint. Training uses an NVIDIA GPU. The paper used CUDA 11.8; see the [official PyTorch 2.0.0 instructions](https://pytorch.org/get-started/previous-versions/#v200) for that build.

## Prepare data

Follow the cache contract in [docs/data.md](docs/data.md). The source data are available through [NeuroBOLT](https://github.com/soupeeli/NeuroBOLT). Download `labram-base.pth` from the [official LaBraM repository](https://github.com/935963004/LaBraM) into `checkpoints/`.

Convert the upstream EEG `.set` / `.fdt` and DiFuMo-64 `.pkl` files into aligned caches:

```bash
pip install -e ".[data]"
python -m seqeft.prepare_data \\
  --eeg-dir /path/to/EEG_set \\
  --bold-dir /path/to/fMRI_difumo64 \\
  --output-dir data/cache/resting_state
```


This release includes the model, training/evaluation code, fixed split and reference configurations. Raw data and the paper's trained weights are not distributed here. Stage 1 produces the VAE checkpoint required by Stage 2.

## Train

```bash
# Stage 1: train scans only, 25 epochs / 800 updates
python -m seqeft.train_vae --config configs/vae.yaml --device cuda

# Stage 2: 150 epochs / 450 updates; exports final EMA weights
python -m seqeft.train --config configs/seqeft.yaml --device cuda

# Additional paper seeds share the same pretrained VAE
python -m seqeft.train --seed 20261013 --output-dir outputs/seqeft_seed20261013
python -m seqeft.train --seed 20261014 --output-dir outputs/seqeft_seed20261014
```

Each epoch saves `last.pt`; the final inference checkpoint is `ema_last.pt`. To resume, use the same config and seed and add `--resume <run-directory>/last.pt`. A fresh run requires a new output directory.

## Evaluate and predict

```bash
python -m seqeft.evaluate \
  --checkpoint outputs/seqeft_seed20261012/ema_last.pt \
  --output evaluation.json --device cuda

python -m seqeft.predict \
  --checkpoint outputs/seqeft_seed20261012/ema_last.pt \
  --eeg example_eeg.npy --output predicted_bold.npy --device cuda
```

Prediction takes preprocessed EEG in microvolts with shape `[512,26,16,200]` and returns normalized BOLD with shape `[512,64]`. A full EMA checkpoint includes the frozen VAE and can run without the original LaBraM/VAE files. Evaluation accepts multiple checkpoint paths to compute the across-seed mean and sample SD.

## Reference results

Resting-state split: 24 training scans from 17 participants and 5 evaluation scans from 5 participants. Results use the fixed epoch-150 EMA checkpoint, averaged across three alignment seeds.

| Tcorr ↑ | fMRI MSE ↓ | FC Corr ↑ | ConnMSE ↓ | F1@25% ↑ | F1@50% ↑ |
|---:|---:|---:|---:|---:|---:|
| 0.4651 ± 0.0025 | 0.2232 ± 0.0026 | 0.6364 ± 0.0099 | 0.0820 ± 0.0024 | 0.5947 ± 0.0086 | 0.7152 ± 0.0048 |

See [reproducibility notes](docs/reproducibility.md) for the protocol and release scope.

## Layout

| Path | Purpose |
|---|---|
| `src/seqeft/models/` | EEG encoders, latent alignment and fMRI VAE |
| `src/seqeft/model.py` | Final paper model factory |
| `src/seqeft/train_vae.py`, `train.py` | Two training stages |
| `src/seqeft/evaluate.py`, `predict.py` | Evaluation and inference |
| `configs/` | Reference configs and fixed split |
| `tests/` | Synthetic model, gradient, data and training checks |

## Acknowledgments and license

We build on [LaBraM](https://github.com/935963004/LaBraM) and [NeuroBOLT](https://github.com/soupeeli/NeuroBOLT), and acknowledge [UnEBOLT](https://github.com/neurdylab/UnEBOLT) for data organization and FC evaluation references.

Code: [GPL-3.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) and [CITATION.cff](CITATION.cff).
