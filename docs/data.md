# 数据准备

本仓库提供上游 EEG `.set` 与 DiFuMo-64 ROI `.pkl` 的转换器，生成训练所需的
EEG–fMRI 缓存。数据本体、受试者原始记录和第三方预训练权重不随代码分发。

数据入口与上游说明：[NeuroBOLT](https://github.com/soupeeli/NeuroBOLT)、
[UnEBOLT](https://github.com/neurdylab/UnEBOLT)。请按照数据提供者的流程获取数据，
并使用下面的缓存合同。当前发布版不包含从原始 MRI 重建或提取 DiFuMo ROI 的流程。

## 从上游文件生成缓存

从仓库根目录运行：

```bash
pip install -e ".[data]"
python -m seqeft.prepare_data \
  --eeg-dir /path/to/EEG_set \
  --bold-dir /path/to/fMRI_difumo64 \
  --output-dir data/cache/resting_state
```

转换器按 `configs/splits/resting_state.yaml` 读取 29 条指定 scan。
每条 scan 需要对应的 `subXX-scanYY_eeg.set`、伴随的 `_eeg.fdt`，以及
`subXX-scanYY_difumo64_roi.pkl`；最后一个文件应包含固定顺序的 64 个 ROI
和末尾的 `global signal clean`、`global signal raw` 两列。
转换器读取这些文件，生成下面的缓存，不修改输入；已存在的 scan 输出不会被覆盖。
已有符合合同的缓存可以直接使用，只需修改配置中的 `data.cache_root`。

## 缓存目录

默认配置读取 `data/cache/resting_state`。每条 scan 的目录中需要四个文件：

| 文件 | 内容 |
| --- | --- |
| `eeg.npy` | `float32 [T,26,16,200]`，EEG 单位为 µV |
| `bold.npy` | `float32 [T,64]`，已完成 scan-level 归一化 |
| `frame_index.npy` | `int64 [T]`，有效 EEG 窗口对应的原始 fMRI 帧号 |
| `metadata.json` | scan、受试者、长度、split 以及 ROI 顺序 |

例如，一条 scan 的路径是
`data/cache/resting_state/sub01-scan01/eeg.npy`。
`metadata.json` 至少包含：

```json
{
  "scan_id": "sub01-scan01",
  "subject_id": "sub01",
  "split": "train",
  "length": 567
}
```

`length` 必须填写该 scan 的实际 T，不能机械使用示例中的 567。
建议同时保存 `bold_columns`（按数组列顺序排列的 64 个 ROI 名称）以核对跨 scan 的 ROI 一致性。
Stage 1 只需 BOLD、帧号和 metadata；Stage 2 还需要 EEG。

## 预处理与对齐

Resting-state 参考实现使用 TR=2.1 秒。每个 BOLD 帧配对采集触发之前的 16 秒 EEG。

1. EEG 重采样到 200 Hz，按下述 26 通道顺序重排；应用 0.5 Hz 高通。
2. 按 fMRI 同步事件 `R149` 分段：`tmin=-16, tmax=0, baseline=(None, None)`；
   在包含端点的整个 epoch 上做基线校正，再读取 µV，保留前 3200 个采样点，
   reshape 为 `[26,16,200]`。
3. BOLD 使用固定顺序的 64 个 DiFuMo ROI，排除 global-signal 列。
   在完整 scan 上先做 5 阶 Butterworth 0.15 Hz 低通（双向 `filtfilt`），再逐 ROI 去均值，
   除以去均值信号绝对值的 95% 分位数加 `1e-8`。
4. 使用 EEG epoch 的原始事件索引选择 BOLD 行，并把同一索引保存为 `frame_index.npy`。

EEG 通道顺序：

```text
FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T7 T8 P7 P8
FPZ FZ CZ PZ POZ OZ FT9 FT10 TP9 TP10
```

缓存仍以 µV 保存 EEG。模型内部使用 `input_scale=0.01`，不要在准备缓存时重复缩放。
BOLD 的滤波与归一化发生在完整 scan 上；训练随机窗口或评价中心窗口不再单独归一化。

## 划分与窗口

`configs/splits/resting_state.yaml` 保存论文使用的固定成员名单：

- 训练：24 scans / 17 位受试者。
- 评价：5 scans / 5 位受试者；与训练受试者无交集。
- 评价扫描：`sub04-scan01`、`sub06-scan01`、`sub15-scan01`、
  `sub16-scan01`、`sub21-scan01`。

YAML 使用 `test` 作为兼容字段。五人评价集在开发过程中被多次使用，
因此论文结果应按 development evaluation 理解。划分文件决定实际集合；
旧缓存 metadata 的标签可以保留，loader 不会重写缓存。

训练时一个 dataset index 对应一条 scan，每次访问随机选择一个连续 512 帧窗口。
Stage 1 每轮有放回抽取 512 个 scan-crop，batch16；
Stage 2 每轮打乱 24 条训练 scan，batch8，共 3 次更新。

评价时每条 scan 只取一个中心窗口：

```python
start = (T - 512) // 2
end = start + 512
```

EEG、BOLD、frame_index 使用同一 `[start:end]`。有效长度至少为 512；
中心评价要求帧号连续。缓存的 `eeg.npy` 不应混入 BOLD、标签或未来预测结果。

## 指标

`seqeft.metrics.scan_metrics` 对整段 `[512,64]` 计算六项指标：

| 字段 | 定义 |
| --- | --- |
| `tcorr` | 各 ROI 时间 Pearson 的均值 |
| `pixcorr` | FC Corr：两张 ROI Pearson 矩阵严格上三角边向量的 Pearson |
| `conn_mse` | 同一上三角边向量的 MSE |
| `f1_025`、`f1_05` | 有符号边值最强 25% / 50% 的 F1 |
| `bold_mse` | 全部 BOLD 元素的 MSE |

F1 沿用上游的百分位数阈值与 `>=` 比较；不使用绝对相关值排序或 exact top-k。
默认的五条评价 scan 每人恰好一条，因此 scan 均值与受试者等权均值相同。
评价命令输出 scan 均值；对 `--split train`，这不等于受试者等权均值，
因为部分训练受试者有多条 scan。需要后一口径时可使用 `aggregate_metrics(rows)`。
论文三 seed 表格对三个 subject-macro 结果计算均值及样本标准差（`ddof=1`）。
常数 ROI、非有限输入和退化 FC 向量会报错，避免把无效相关系数当作正常分数。
