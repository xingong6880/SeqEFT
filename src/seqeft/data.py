"""SeqEFT 的只读缓存与固定 512 帧数据集。

一个 dataset index 对应一条 scan。训练时随机选择连续窗口；评估时只取
每条 scan 的中心窗口。EEG、BOLD 与帧号始终用同一位置裁剪。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, get_worker_info

WINDOW_LENGTH = 512
ROI_COUNT = 64
EEG_SHAPE = (26, 16, 200)


@dataclass(frozen=True)
class CachedBOLDScan:
    """只保存缓存路径和身份；数组以只读 mmap 方式加载。"""

    root: Path
    bold_path: Path
    frame_index_path: Path
    subject_id: str
    scan_id: str
    split: str
    length: int
    cache_version: object = None


def load_split_manifest(path) -> dict:
    """读取划分并拒绝重复 scan 或跨集合的同一受试者。"""
    with Path(path).open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    names = set(manifest) if isinstance(manifest, dict) else set()
    if names not in ({"train", "test"}, {"train", "val", "test"}):
        raise ValueError("split 必须包含 train/test，或 train/val/test")
    seen_scans, seen_subjects = set(), set()
    for split in ("train", "val", "test"):
        if split not in manifest:
            continue
        scans = manifest[split]
        if not isinstance(scans, list) or not all(isinstance(x, str) for x in scans):
            raise ValueError("split 中必须是 scan_id 字符串列表")
        if len(set(scans)) != len(scans) or seen_scans.intersection(scans):
            raise ValueError("split 包含重复 scan")
        subjects = {scan.split("-", 1)[0] for scan in scans}
        if seen_subjects.intersection(subjects):
            raise ValueError("同一受试者不能出现在多个 split")
        seen_scans.update(scans)
        seen_subjects.update(subjects)
    return manifest


def load_existing_bold_caches(config, split="train"):
    """按配置读取缓存身份；兼容字典配置与旧版 namespace 配置。

    两路划分可以复用旧缓存的 train/val/test 标签，实际成员以 split YAML
    为准。这个映射只改变内存中的 handle，不会修改缓存。
    """
    data = config["data"] if isinstance(config, dict) else config.data
    field = lambda name: data[name] if isinstance(data, dict) else getattr(data, name)
    manifest = load_split_manifest(field("split_file"))
    if split not in manifest:
        raise ValueError("所请求 split 不在划分文件中")
    cache_root = Path(field("cache_root"))
    caches = []
    for scan_id in manifest[split]:
        root = cache_root / scan_id
        with (root / "metadata.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        subject_id = scan_id.split("-", 1)[0]
        if metadata["scan_id"] != scan_id or metadata["subject_id"] != subject_id:
            raise ValueError("缓存 scan/subject 身份与 split 不一致: " + scan_id)
        cache_split = metadata["split"]
        if cache_split not in {"train", "val", "test"}:
            raise ValueError("缓存 split 标签无效: " + scan_id)
        if "val" in manifest and cache_split != split:
            raise ValueError("缓存 split 与三路 manifest 不一致: " + scan_id)
        cache = CachedBOLDScan(
            root, root / "bold.npy", root / "frame_index.npy",
            subject_id, scan_id, split, int(metadata["length"]),
            metadata.get("cache_version"),
        )
        # 与原始 loader 一样，先检查文件可读及 metadata.length 一致。
        bold = np.load(cache.bold_path, mmap_mode="r", allow_pickle=False)
        np.load(cache.frame_index_path, mmap_mode="r", allow_pickle=False)
        if bold.shape[0] != cache.length:
            raise ValueError("缓存 length 与数组不一致: " + scan_id)
        caches.append(cache)
    return caches


def load_existing_paired_caches(config, split="train"):
    """EEG 与 BOLD 共用同一份 scan handle 和划分。"""
    return load_existing_bold_caches(config, split=split)


def contiguous_runs(frame_index):
    """返回连续帧号对应的数组位置区间 [start, end)。"""
    frames = np.asarray(frame_index)
    if frames.ndim != 1:
        raise ValueError("frame_index 必须是一维数组")
    if not frames.size:
        return []
    points = [0] + (np.flatnonzero(np.diff(frames) != 1) + 1).tolist() + [len(frames)]
    return list(zip(points[:-1], points[1:]))


def valid_window_starts(frame_index, window_length=512):
    """枚举完整且不跨越帧号断点的窗口起点。"""
    if isinstance(window_length, bool) or not isinstance(window_length, int) or window_length <= 0:
        raise ValueError("window_length 必须是正整数")
    return [
        start
        for first, end in contiguous_runs(frame_index)
        for start in range(first, end - window_length + 1)
    ]


class _WindowDataset(Dataset):
    """共享只读数组、scan 身份及同步裁剪逻辑。"""

    def __init__(self, caches: Sequence, *, paired: bool):
        if not caches:
            raise ValueError("至少需要一个 scan cache")
        self.paired = paired
        self._records = []
        for cache in caches:
            names = ("bold", "frame_index", "eeg") if paired else ("bold", "frame_index")
            arrays = {}
            for name in names:
                if hasattr(cache, name):
                    arrays[name] = np.asarray(getattr(cache, name))
                else:
                    filename = "frame_index.npy" if name == "frame_index" else name + ".npy"
                    path = getattr(cache, name + "_path", Path(cache.root) / filename)
                    arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            bold, frames = arrays["bold"], arrays["frame_index"]
            if bold.ndim != 2 or bold.shape[1] != ROI_COUNT:
                raise ValueError("BOLD 必须为 [T,64]")
            if frames.shape != (len(bold),) or not np.issubdtype(frames.dtype, np.integer):
                raise ValueError("frame_index 必须为同长度的一维整数数组")
            if not np.isfinite(bold).all():
                raise ValueError("BOLD 必须全部为有限值")
            if getattr(cache, "length", len(bold)) != len(bold):
                raise ValueError("缓存 length 与数组不一致")
            if paired:
                eeg = arrays["eeg"]
                if eeg.shape != (len(bold),) + EEG_SHAPE or not np.isfinite(eeg).all():
                    raise ValueError("EEG 必须为有限的 [T,26,16,200] 数组")
            starts = tuple(valid_window_starts(frames))
            if not starts:
                raise ValueError("scan {} 不包含完整的连续 512 帧窗口".format(cache.scan_id))
            self._records.append((cache, arrays, starts))

    def __len__(self):
        return len(self._records)

    @property
    def valid_starts(self):
        return tuple(record[2] for record in self._records)

    @property
    def scan_ids(self):
        return tuple(str(record[0].scan_id) for record in self._records)

    def _sample(self, index, start):
        cache, arrays, starts = self._records[index]
        if start not in starts:
            raise ValueError("窗口起点不合法")
        sample = {
            key: torch.from_numpy(np.array(value[start:start + WINDOW_LENGTH],
                dtype=np.int64 if key == "frame_index" else np.float32, copy=True))
            for key, value in arrays.items()
        }
        sample.update(
            scan_id=str(cache.scan_id), subject_id=str(cache.subject_id),
            split=str(cache.split), start=int(start),
        )
        return sample


class _RandomWindowDataset(_WindowDataset):
    """保留论文实现的 seed/epoch/worker 混合规则及 NumPy choice 顺序。"""

    def __init__(self, caches, *, seed, paired):
        super().__init__(caches, paired=paired)
        self.seed = int(seed)
        self._epoch = 0
        self._rngs = {}

    def set_epoch(self, epoch):
        self._epoch = int(epoch)
        self._rngs.clear()

    def _rng(self):
        worker = get_worker_info()
        worker_key = None if worker is None else worker.id
        if worker_key not in self._rngs:
            worker_seed = self.seed if worker is None else int(worker.seed)
            seed = (worker_seed + self.seed + self._epoch * 1000003) % (2**63 - 1)
            self._rngs[worker_key] = np.random.default_rng(seed)
        return self._rngs[worker_key]

    def __getitem__(self, index):
        start = int(self._rng().choice(self._records[index][2]))
        return self._sample(index, start)


class RandomPairedWindowDataset(_RandomWindowDataset):
    """Stage 2：每次访问一条 scan，随机裁剪配对 EEG/BOLD。"""

    def __init__(self, caches, *, seed=20261012):
        super().__init__(caches, seed=seed, paired=True)


class RandomFixedWindowBOLDDataset(_RandomWindowDataset):
    """Stage 1：只读取 BOLD，保持一个 scan 对应一个 dataset index。"""

    def __init__(self, caches, *, window_length=512, roi_count=64, seed=2026):
        if window_length != 512 or roi_count != 64:
            raise ValueError("公开配置固定 window_length=512、roi_count=64")
        self.window_length, self.roi_count = window_length, roi_count
        super().__init__(caches, seed=seed, paired=False)


class CenterPairedWindowDataset(_WindowDataset):
    """每条 scan 取唯一连续序列的中心 512 帧，拒绝跨断点评估。"""

    def __init__(self, caches):
        super().__init__(caches, paired=True)
        self._starts = []
        for _, arrays, valid in self._records:
            runs = contiguous_runs(arrays["frame_index"])
            if len(runs) != 1:
                raise ValueError("中心窗口要求整条 scan 的 frame_index 连续")
            first, end = runs[0]
            start = first + (end - first - WINDOW_LENGTH) // 2
            if start not in valid:
                raise ValueError("中心窗口不足 512 帧")
            self._starts.append(start)

    @property
    def starts(self):
        return tuple(zip(self.scan_ids, self._starts))

    def __getitem__(self, index):
        return self._sample(index, self._starts[index])


def paired_collate(samples):
    """将同一字段堆叠为 batch，字符串身份仍保留为列表。"""
    if not samples:
        raise ValueError("不能 collate 空列表")
    batch = {}
    for key in ("eeg", "bold", "frame_index", "scan_id", "subject_id", "split", "start"):
        values = [sample[key] for sample in samples]
        if key in ("eeg", "bold", "frame_index"):
            batch[key] = torch.stack(values)
        elif key == "start":
            batch[key] = torch.as_tensor(values, dtype=torch.long)
        else:
            batch[key] = values
    return batch
