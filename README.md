# SeqEFT

**Latent Sequence Alignment for EEG-to-fMRI Translation**

[English](README.en.md) · [项目主页](https://xingong6880.github.io/projects/seqeft/) · [数据准备](docs/data.md) · [复现说明](docs/reproducibility.md)

SeqEFT 从 EEG 重建多脑区 fMRI（BOLD）时间序列。它先学习 fMRI 的潜在序列表示，再训练 EEG 编码器与这些表示对齐，最后通过冻结的 fMRI 解码器重建 64 个 ROI 的信号。

## 方法

1. **fMRI 表示预训练**：序列 VAE 将 `[512,64]` 的 fMRI 序列压缩到 `[256,16]`，学习跨时间、跨脑区的表示。
2. **EEG 潜在序列对齐**：LaBraM 与多尺度频谱编码器提取 EEG 特征，内容相关的局部聚合与时序混合预测 fMRI latent；冻结的解码器生成 BOLD 序列。

训练目标为 latent MSE + 0.1 × ConnMSE，推理只需要 EEG。默认模型联合处理 512 个 fMRI 时间点，每个时间点对应之前 16 秒的 26 通道 EEG。

## 快速开始

参考环境：Python 3.9、PyTorch 2.0.0。所有命令在仓库根目录执行。

```bash
git clone https://github.com/xingong6880/SeqEFT.git
cd SeqEFT
conda create -n seqeft python=3.9 -y
conda activate seqeft
pip install -e ".[test]"

# 无需数据或预训练权重：验证形状、梯度和模型组件
python -m seqeft.smoke
python -m pytest -q
```

正式训练建议使用 NVIDIA GPU。论文环境使用 CUDA 11.8；需要这一环境时，先按 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/#v200) 安装对应的 PyTorch 2.0.0，再安装本项目。

## 数据与预训练权重

按 [数据准备文档](docs/data.md) 放置已对齐的 EEG/BOLD 缓存，并从 [LaBraM 官方仓库](https://github.com/935963004/LaBraM) 下载 `labram-base.pth` 到 `checkpoints/`。数据来源见 [NeuroBOLT](https://github.com/soupeeli/NeuroBOLT)。

上游 EEG `.set` / `.fdt` 和 DiFuMo-64 `.pkl` 可直接转换为训练缓存：

```bash
pip install -e ".[data]"
python -m seqeft.prepare_data \\
  --eeg-dir /path/to/EEG_set \\
  --bold-dir /path/to/fMRI_difumo64 \\
  --output-dir data/cache/resting_state
```


本版本发布模型、训练与评估代码、固定划分和参考配置。原始数据与论文训练权重不随仓库分发；Stage 1 会生成 Stage 2 所需的 VAE checkpoint。默认配置中的路径均可编辑。

## 训练

```bash
# Stage 1：25 epochs，800 次参数更新，仅使用训练扫描
python -m seqeft.train_vae --config configs/vae.yaml --device cuda

# Stage 2：150 epochs，450 次参数更新，输出 EMA 推理权重
python -m seqeft.train --config configs/seqeft.yaml --device cuda
```

第二阶段三个参考 seed 为 `20261012`、`20261013`、`20261014`，共享同一个 Stage 1 VAE：

```bash
python -m seqeft.train --seed 20261013 --output-dir outputs/seqeft_seed20261013
python -m seqeft.train --seed 20261014 --output-dir outputs/seqeft_seed20261014
```

每轮保存 `last.pt`，训练结束保存 `ema_last.pt`。恢复训练时使用相同配置和 seed，追加 `--resume <该次运行的 last.pt>`。新训练要求输出目录尚不存在，防止覆盖已有实验。

## 评估与推理

```bash
# 单个 seed 的 Center512 评估；传入多个 checkpoint 可汇总 mean ± sample SD
python -m seqeft.evaluate \
  --checkpoint outputs/seqeft_seed20261012/ema_last.pt \
  --output evaluation.json --device cuda

# 输入已预处理、以微伏为单位的 EEG 数组 [512,26,16,200]
python -m seqeft.predict \
  --checkpoint outputs/seqeft_seed20261012/ema_last.pt \
  --eeg example_eeg.npy --output predicted_bold.npy --device cuda
```

输出为归一化的 `[512,64]` BOLD 序列。完整 EMA checkpoint 已包含冻结的 VAE，推理时无需再次提供 LaBraM 或 VAE 的独立文件。

## 参考结果

静息态数据：24 个训练扫描 / 17 位参与者，5 个评价扫描 / 5 位参与者。以下为固定 epoch 150 的 EMA 模型，三个 seed 的均值 ± 样本标准差。

| Tcorr ↑ | fMRI MSE ↓ | FC Corr ↑ | ConnMSE ↓ | F1@25% ↑ | F1@50% ↑ |
|---:|---:|---:|---:|---:|---:|
| 0.4651 ± 0.0025 | 0.2232 ± 0.0026 | 0.6364 ± 0.0099 | 0.0820 ± 0.0024 | 0.5947 ± 0.0086 | 0.7152 ± 0.0048 |

复现所需的固定预算、checkpoint 选择、数据边界和验证范围见 [复现说明](docs/reproducibility.md)。

## 代码导航

| 路径 | 用途 |
|---|---|
| `src/seqeft/models/` | EEG 编码器、局部对齐、时序模块、fMRI VAE |
| `src/seqeft/model.py` | 论文模型构造入口 |
| `src/seqeft/train_vae.py` / `train.py` | 两阶段训练 |
| `src/seqeft/evaluate.py` / `predict.py` | 六指标评估与 EEG 推理 |
| `configs/` | 可移植参考配置与固定扫描划分 |
| `tests/` | 合成输入下的模型、梯度、数据与训练检查 |

## 致谢与许可

EEG 编码器基于 [LaBraM](https://github.com/935963004/LaBraM) 和 [NeuroBOLT](https://github.com/soupeeli/NeuroBOLT) 的工作；同时感谢 [UnEBOLT](https://github.com/neurdylab/UnEBOLT) 的数据组织与功能连接评估参考。

代码采用 [GPL-3.0](LICENSE)，上游说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。引用软件可使用 [CITATION.cff](CITATION.cff)。
