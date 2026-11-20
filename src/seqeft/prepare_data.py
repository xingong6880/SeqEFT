"""把上游 resting-state EEG .set 与 DiFuMo-64 .pkl 转换为 SeqEFT 缓存。

运行前安装 pip install -e '.[data]'。在仓库根目录执行：
python -m seqeft.prepare_data --eeg-dir EEG_set --bold-dir fMRI_difumo64 \
    --output-dir data/cache/resting_state

仅处理 configs/splits/resting_state.yaml 中的 29 条扫描；不下载数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .data import load_split_manifest


VU_CHANNELS = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "FPZ", "FZ", "CZ", "PZ", "POZ", "OZ", "FT9", "FT10", "TP9", "TP10",
)
_NON_EEG_CHANNELS = {
    "EOG1", "EOG2", "EMG1", "EMG2", "EMG3", "ECG", "CWL1", "CWL2", "CWL3", "CWL4",
}
_GLOBAL_SIGNAL_COLUMNS = ("global signal clean", "global signal raw")
SPLIT_FILE = Path("configs/splits/resting_state.yaml")


def _epoch_eeg(raw):
    """复用原始预处理顺序，返回微伏 EEG 与有效 R149 事件的原始索引。

    MNE 在包含两个端点的 3201 点 epoch 上做全窗 baseline，再保留前
    3200 点。此处不除以 100；模型 tokenizer 内部负责 input_scale=0.01。
    """
    import mne

    if raw.info["sfreq"] != 200:
        raw.load_data()
        raw.resample(200)
    raw.rename_channels({name: name.upper().replace("'", "") for name in raw.ch_names})
    dropped = [name for name in raw.ch_names if name in _NON_EEG_CHANNELS]
    if dropped:
        raw.drop_channels(dropped)
    raw.pick_channels(list(VU_CHANNELS), ordered=True)
    events, event_id = mne.events_from_annotations(raw)
    if "R149" not in event_id:
        raise ValueError("EEG 缺少 fMRI 同步事件 R149")
    events = mne.pick_events(events, include=event_id["R149"])
    # 原始文件重采样时已经 preload；显式 load 同时兼容原本就是 200 Hz 的文件。
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=None)
    epochs = mne.Epochs(
        raw, events, event_id={"R149": event_id["R149"]},
        tmin=-16.0, tmax=0.0, baseline=(None, None),
        reject=None, preload=True,
    )
    eeg = epochs.get_data(units="uV")[:, :, :3200]
    eeg = eeg.reshape(eeg.shape[0], 26, 16, 200).astype(np.float32, copy=False)
    return eeg, epochs.selection.astype(np.int64)


def _validate_bold_schema(frame):
    """保持文件中的 64 ROI 顺序，排除末尾两列 global signal。"""
    columns = tuple(frame.columns)
    if len(columns) != 66 or columns[-2:] != _GLOBAL_SIGNAL_COLUMNS:
        raise ValueError("BOLD 必须为 64 ROI + global signal clean/raw，共 66 列")
    roi_columns = columns[:64]
    if any(not isinstance(name, str) or not name.strip() for name in roi_columns):
        raise ValueError("ROI 列名必须是非空字符串")
    if len(set(roi_columns)) != 64:
        raise ValueError("64 个 ROI 列名必须唯一")
    return roi_columns


def normalize_bold(bold):
    """逐 ROI 在完整 scan 上去均值并按绝对值 95% 分位数归一化。

    输入与输出均为 [64,T]；在选取 EEG 对应帧之前调用。
    """
    means = np.mean(bold, axis=1, keepdims=True)
    centered = bold - means
    scale = np.quantile(np.abs(centered), q=0.95, axis=1, keepdims=True) + 1e-8
    return centered / scale


def preprocess_bold(frame, selection):
    """执行原始 5 阶 0.15 Hz 低通、全 scan 归一化与事件索引对齐。"""
    from scipy.signal import butter, filtfilt

    columns = _validate_bold_schema(frame)
    bold = frame.loc[:, columns].to_numpy().T
    selection = np.asarray(selection)
    if selection.ndim != 1 or not np.issubdtype(selection.dtype, np.integer):
        raise ValueError("BOLD selection 必须是一维整数数组")
    if len(selection) and (selection.min() < 0 or selection.max() >= bold.shape[1]):
        raise ValueError("EEG 事件索引超出 BOLD 帧范围")
    fs = 1.0 / 2.1
    b, a = butter(N=5, Wn=0.15 / (0.5 * fs), btype="low", analog=False)
    filtered = filtfilt(b, a, bold, axis=1)
    selected = normalize_bold(filtered)[:, selection].T
    return selected.astype(np.float32, copy=False), columns


def _write_scan(output_dir, scan_id, split, eeg, bold, frame_index, columns):
    """在临时目录写完整四个文件，再发布 scan 目录；不覆盖已有缓存。"""
    target = output_dir / scan_id
    if target.exists():
        raise FileExistsError(target)
    if eeg.shape != (len(frame_index), 26, 16, 200) or bold.shape != (len(frame_index), 64):
        raise ValueError("EEG/BOLD/frame_index 形状或长度不一致")
    if not len(frame_index) or not np.isfinite(eeg).all() or not np.isfinite(bold).all():
        raise ValueError("预处理结果必须非空且全部为有限值")
    metadata = {
        "cache_version": 2, "scan_id": scan_id,
        "subject_id": scan_id.split("-", 1)[0], "split": split,
        "length": int(len(frame_index)),
        "eeg_shape": list(eeg.shape), "bold_shape": list(bold.shape),
        "frame_index_shape": list(frame_index.shape),
        "eeg_dtype": str(eeg.dtype), "bold_dtype": str(bold.dtype),
        "frame_index_dtype": str(frame_index.dtype),
        "bold_columns": list(columns),
        "bold_schema_fingerprint": hashlib.sha256(
            json.dumps(list(columns), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "preprocessing": {
            "tr_seconds": 2.1, "eeg_sampling_rate": 200, "eeg_units": "uV",
            "eeg_highpass_hz": 0.5, "eeg_baseline": "entire inclusive [-16,0] epoch",
            "eeg_crop_samples": 3200, "model_input_scale": 0.01,
            "bold_lowpass_hz": 0.15, "bold_filter_order": 5,
            "bold_normalization": "whole-scan demean / (q95(abs(demeaned)) + 1e-8)",
        },
    }
    staging = Path(tempfile.mkdtemp(prefix="." + scan_id + "-", dir=output_dir))
    try:
        np.save(staging / "eeg.npy", eeg)
        np.save(staging / "bold.npy", bold)
        np.save(staging / "frame_index.npy", frame_index)
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prepare_data(eeg_dir, bold_dir, output_dir):
    """只读取固定名单中的源文件；全部路径先检查，再逐条生成缓存。"""
    import mne
    import pandas as pd

    eeg_dir, bold_dir, output_dir = (Path(p).expanduser().resolve()
                                     for p in (eeg_dir, bold_dir, output_dir))
    if not eeg_dir.is_dir() or not bold_dir.is_dir():
        raise FileNotFoundError("EEG 或 BOLD 输入目录不存在")
    if output_dir.is_relative_to(eeg_dir) or output_dir.is_relative_to(bold_dir):
        raise ValueError("输出目录必须位于两个源数据目录之外")
    manifest = load_split_manifest(SPLIT_FILE)
    if set(manifest) != {"train", "test"} or len(manifest["train"]) != 24 or len(manifest["test"]) != 5:
        raise ValueError("需要仓库提供的固定 24/5 resting-state 划分")
    sources = []
    for split in ("train", "test"):
        for scan_id in manifest[split]:
            eeg_path = eeg_dir / (scan_id + "_eeg.set")
            bold_path = bold_dir / (scan_id + "_difumo64_roi.pkl")
            for path in (eeg_path, eeg_path.with_suffix(".fdt"), bold_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if (output_dir / scan_id).exists():
                raise FileExistsError(output_dir / scan_id)
            sources.append((scan_id, split, eeg_path, bold_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_order = None
    for scan_id, split, eeg_path, bold_path in sources:
        raw = mne.io.read_raw_eeglab(str(eeg_path), preload=False)
        eeg, selection = _epoch_eeg(raw)
        bold, columns = preprocess_bold(pd.read_pickle(bold_path), selection)
        if roi_order is None:
            roi_order = columns
        if columns != roi_order:
            raise ValueError("跨 scan 的 ROI 顺序不一致: " + scan_id)
        _write_scan(output_dir, scan_id, split, eeg, bold, selection, columns)
        print(f"{scan_id}: {len(selection)} aligned frames", flush=True)
        del raw, eeg, bold
    return {"scans": len(sources), "output_dir": str(output_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg-dir", required=True, help="上游 EEG_set 目录，包含 .set 和伴生 .fdt")
    parser.add_argument("--bold-dir", required=True, help="上游 fMRI_difumo64 目录")
    parser.add_argument("--output-dir", required=True, help="新缓存目录，不覆盖已有 scan")
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.eeg_dir, args.bold_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
