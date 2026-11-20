# SeqEFT

**Latent Sequence Alignment for EEG-to-fMRI Translation**

SeqEFT reconstructs 512-frame fMRI (BOLD) time series for 64 brain regions from EEG. It aligns EEG features with the latent sequence of an fMRI VAE and uses the frozen VAE decoder to reconstruct BOLD. [Project page](https://xingong6880.github.io/projects/seqeft/)

## Quick start

Use Python 3.9 and run the commands from the repository root. The reference GPU setup uses PyTorch 2.0.0 with CUDA 11.8. [PyTorch builds](https://pytorch.org/get-started/previous-versions/#v200)

```bash
python -m pip install -e ".[data,test]"
python -m pytest -q
```

## Train and evaluate

Download `EEG_set/` and `fMRI_difumo64/` from the NeuroBOLT dataset and place `labram-base.pth` in `checkpoints/`. Train the VAE once and reuse it for the three alignment runs. Evaluation reports the mean and sample standard deviation across seeds on the fixed development split. [NeuroBOLT dataset](https://huggingface.co/datasets/NeurdyLab/NeuroBOLT/tree/main) · [LaBraM base weights](https://github.com/935963004/LaBraM/blob/main/checkpoints/labram-base.pth) · [Fixed split](configs/splits/resting_state.yaml)

```bash
python -m seqeft.prepare_data \
  --eeg-dir /path/to/EEG_set \
  --bold-dir /path/to/fMRI_difumo64 \
  --output-dir data/cache/resting_state

python -m seqeft.train_vae
for seed in 20261012 20261013 20261014; do
  python -m seqeft.train --seed "$seed" --output-dir "outputs/seqeft_seed${seed}"
done
python -m seqeft.evaluate \
  --checkpoint outputs/seqeft_seed{20261012,20261013,20261014}/ema_last.pt

python -m seqeft.predict \
  --checkpoint outputs/seqeft_seed20261012/ema_last.pt \
  --eeg example_eeg.npy --output predicted_bold.npy
```

For prediction, use a contiguous 512-frame window from the preprocessed `eeg.npy`. The input is in microvolts with shape `[512, 26, 16, 200]`. The output is normalized BOLD with shape `[512, 64]`.

## License and citation

SeqEFT uses GPL-3.0 and adapts EEG encoding and preprocessing from NeuroBOLT by Yamin Li and contributors. It uses a LaBraM temporal backbone and follows UnEBOLT's data and evaluation conventions. [GPL-3.0](LICENSE) · [NeuroBOLT reference](https://github.com/soupeeli/NeuroBOLT/commit/b51b966516a9b0c0a87e6ee6a7b997101a443394) · [LaBraM MIT notice](LICENSES/LaBraM-MIT.txt) · [UnEBOLT reference](https://github.com/neurdylab/UnEBOLT/commit/09a382897b85820854ddd4e47a1f851fa43d32d9)

Cite Junwen Gong, *SeqEFT: Latent Sequence Alignment for EEG-to-fMRI Translation*, v0.1.0 (2026).
