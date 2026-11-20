# 复现与版本范围 / Reproducibility

## 参考实验

本版本整理论文的 **512 帧静息态 SeqEFT 主模型**，包含预处理缓存接口、两个训练阶段、
EEG 推理、Center512 评价和固定扫描划分。模型输出的是 64 个 DiFuMo ROI 的归一化
BOLD 时间序列。

NIH 256 帧迁移、不同长度的消融和各 baseline 的独立训练程序属于原研究实验，
未打包在这个主模型版本中。上游 baseline 实现见 README 的致谢链接。

## 固定训练配置

| 项目 | Stage 1：fMRI VAE | Stage 2：EEG 对齐 |
|---|---|---|
| 随机种子 | 2026 | 20261012 / 20261013 / 20261014 |
| 输入 / 输出 | `[B,512,64]` → latent `[B,256,16]` | EEG `[B,512,26,16,200]` → BOLD `[B,512,64]` |
| Batch size | 16 | 8 |
| 每轮采样 | 有放回抽取 512 个 scan-crop | 每条训练 scan 一个随机 crop，共 24 个 |
| 训练预算 | 25 epochs / 800 updates | 150 epochs / 450 updates |
| AdamW 学习率 | 0.001 | LaBraM 1e-5；其余 EEG 编码器 1e-4；对齐模块 3e-4 |
| Weight decay | 0.01 | 0.01 |
| Schedule | 恒定 | 前 5 轮 warm-up，之后 cosine 到 0 |
| 损失 | MSE + 0.1 spectral + 0.001 KL | latent MSE + 0.1 ConnMSE |
| 最终模型 | 第 25 轮 | 第 150 轮 EMA，decay=0.99 |

三个 EEG alignment seed 共用一个 Stage 1 VAE。跨 seed 的 SD 因而反映在固定 VAE
下第二阶段的训练变化。

训练使用 FP32、`torch.set_float32_matmul_precision("high")`，默认
`frame_chunk_size=16` 是 EEG 编码计算批次，不改变模型的 512 帧上下文。
VAE 始终冻结并处于 eval 模式；decoder 对输入 latent 的梯度仍用于 FC 损失。
EMA 从第一次 optimizer 更新之后开始，不平均冻结参数或 buffers。

Stage 1 的 crop epoch 从 1 计数，Stage 2 从 0 计数，保留原始随机数序列。
两个阶段均保存 optimizer 与 RNG 状态以支持同配置恢复。

## 数据与评价

- 固定划分是 24 个训练扫描 / 17 位参与者，以及 5 个评价扫描 / 5 位参与者。
- EEG/BOLD 在完整 scan 上预处理，然后按同步帧号裁剪；中心窗口不重新归一化。
- 评价窗口起点为 `(T - 512) // 2`，要求有效帧号连续。
- 使用固定训练终点的 EMA checkpoint；本发布版不在评价集上选择 best epoch。
- 五人评价集在研究开发期间被多次使用，应按 development evaluation 理解。
- 每个评价 scan 计算六项指标，先在 seed 内平均，再计算三个 seed 的均值和
  样本 SD（`ddof=1`）。该评价划分每位参与者只有一个 scan，所以 scan mean
  与 subject mean 相同。
- FC 使用严格上三角的 2,016 条边；F1 对有符号相关值分别以 75% / 50% 分位数
  截断，并使用 `>=`，不是绝对值 top-k。

原参考环境为 Python 3.9.25、PyTorch 2.0.0、CUDA 11.8；训练运行在 RTX 3090/4090。
不同设备、驱动或数值库版本可能影响最终训练结果。

## 本次发布的验证

清理包结构后，与参考研究实现进行了以下检查：

- 所有 **587 个初始化 state tensors 完全一致**，模型构造后的 PyTorch RNG 一致。
- 原论文 EMA checkpoint 可 strict-load。
- 单帧真实 EEG 编码，以及完整长度的局部聚合、时序混合、latent head、冻结 decoder
  与 target encoder 输出，最大绝对误差均为 **0.0**。
- 训练数据的随机 crop 顺序、配对数组、中心窗口及六项评价指标与参考实现一致。
- 合成数据上的 VAE 前向/反向、EEG 分支、冻结 decoder 的梯度、EMA、schedule 与
  随机状态恢复由公开测试覆盖。

上述是实现等价性与运行验证；本次发布没有重新运行完整的三 seed 论文训练。

## English release scope

This version contains the reference **512-frame resting-state SeqEFT model**: data-cache
preparation, VAE pretraining, EEG latent alignment, inference, fixed scan split, and six-metric
Center512 evaluation. NIH transfer, sequence-length ablations and baseline training runners
are outside this focused release.

The public model was checked against the research implementation: all 587 initial state tensors
and post-construction RNG match exactly; a real paper EMA checkpoint loads strictly; the tested
EEG encoder and full-length latent/decoder outputs have zero numerical difference. Data crops
and metric definitions also match. These checks verify the implementation; the full three-seed
training experiment was not rerun for this release.

Data and paper-trained checkpoints are acquired separately. The evaluation participants were
used during method development. Results use the fixed final epoch and report sample SD across
alignment seeds, sharing one pretrained VAE.
