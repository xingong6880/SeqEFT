# Third-party notices

SeqEFT is distributed under GNU GPL version 3; see [LICENSE](LICENSE).
The research implementation was reorganized into the `seqeft` package on 2026-11-20.
The release preserves the model computations and checkpoint parameter names, provides portable
configuration and standalone commands, and retains attribution for the work it builds on.

## NeuroBOLT

- Project: [soupeeli/NeuroBOLT](https://github.com/soupeeli/NeuroBOLT)
- Research reference commit: `b51b966516a9b0c0a87e6ee6a7b997101a443394`
- Authors: Yamin Li and the NeuroBOLT contributors.
- License: GNU GPL version 3; full upstream text is retained in
  [LICENSES/NeuroBOLT-GPL-3.0.txt](LICENSES/NeuroBOLT-GPL-3.0.txt).
- Relationship: EEG spatiotemporal and multi-scale spectral architecture, data-window
  construction and reference preprocessing. Local adapted implementations appear in
  `src/seqeft/models/labram.py`, `spectral.py`, the EEG tokenizer and data preparation.

## LaBraM

- Project: [935963004/LaBraM](https://github.com/935963004/LaBraM)
- License: MIT; original copyright and permission notice are retained in
  [LICENSES/LaBraM-MIT.txt](LICENSES/LaBraM-MIT.txt).
- Relationship: pretrained EEG temporal backbone and checkpoint format. The standalone
  implementation preserves checkpoint-compatible parameter names.
- Pretrained weights are downloaded from the upstream project and are not bundled here.

## UnEBOLT

- Project: [neurdylab/UnEBOLT](https://github.com/neurdylab/UnEBOLT)
- Research reference commit: `09a382897b85820854ddd4e47a1f851fa43d32d9`
- Relationship: multi-ROI data organization and functional-connectivity evaluation conventions.
- Its independent baseline runner is not vendored in this release.

## Installed dependencies

PyTorch, NumPy, PyYAML, einops, linear-attention-transformer and optional preprocessing
packages retain their respective licenses. They are installed through package dependencies;
their source is not vendored here.

Dataset access and pretrained-weight terms remain those of the original providers.
