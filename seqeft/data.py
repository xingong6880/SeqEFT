"""Read-only scan caches and synchronized 512-frame windows."""
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
    """Scan identity and paths to memory-mapped arrays."""

    root: Path
    bold_path: Path
    frame_index_path: Path
    subject_id: str
    scan_id: str
    split: str
    length: int
    cache_version: object = None


def load_split_manifest(path) -> dict:
    """Load a split with unique scans and participant-disjoint sets."""
    with Path(path).open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    names = set(manifest) if isinstance(manifest, dict) else set()
    if names not in ({"train", "test"}, {"train", "val", "test"}):
        raise ValueError("The manifest must contain train and test splits, with optional val")
    seen_scans, seen_subjects = set(), set()
    for split in ("train", "val", "test"):
        if split not in manifest:
            continue
        scans = manifest[split]
        if not isinstance(scans, list) or not all(isinstance(x, str) for x in scans):
            raise ValueError("Each split must contain a list of scan ID strings")
        if len(set(scans)) != len(scans) or seen_scans.intersection(scans):
            raise ValueError("The manifest contains a duplicate scan")
        subjects = {scan.split("-", 1)[0] for scan in scans}
        if seen_subjects.intersection(subjects):
            raise ValueError("The same participant appears in multiple splits")
        seen_scans.update(scans)
        seen_subjects.update(subjects)
    return manifest


def load_existing_bold_caches(config, split="train"):
    """Load cache handles from a mapping or namespace config.

    Two-way manifests may reuse caches from older train/val/test splits.
    """
    data = config["data"] if isinstance(config, dict) else config.data
    field = lambda name: data[name] if isinstance(data, dict) else getattr(data, name)
    manifest = load_split_manifest(field("split_file"))
    if split not in manifest:
        raise ValueError("The requested split is missing from the manifest")
    cache_root = Path(field("cache_root"))
    caches = []
    for scan_id in manifest[split]:
        root = cache_root / scan_id
        with (root / "metadata.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        subject_id = scan_id.split("-", 1)[0]
        if metadata["scan_id"] != scan_id or metadata["subject_id"] != subject_id:
            raise ValueError("Cache identity does not match the manifest: " + scan_id)
        cache_split = metadata["split"]
        if cache_split not in {"train", "val", "test"}:
            raise ValueError("Invalid cache split label: " + scan_id)
        if "val" in manifest and cache_split != split:
            raise ValueError("Cache split does not match the three-way manifest: " + scan_id)
        cache = CachedBOLDScan(
            root, root / "bold.npy", root / "frame_index.npy",
            subject_id, scan_id, split, int(metadata["length"]),
            metadata.get("cache_version"),
        )
        bold = np.load(cache.bold_path, mmap_mode="r", allow_pickle=False)
        np.load(cache.frame_index_path, mmap_mode="r", allow_pickle=False)
        if bold.shape[0] != cache.length:
            raise ValueError("Cache length does not match the BOLD array: " + scan_id)
        caches.append(cache)
    return caches


def load_existing_paired_caches(config, split="train"):
    return load_existing_bold_caches(config, split=split)


def contiguous_runs(frame_index):
    """Return half-open array intervals with consecutive frame indices."""
    frames = np.asarray(frame_index)
    if frames.ndim != 1:
        raise ValueError("frame_index must be one-dimensional")
    if not frames.size:
        return []
    points = [0] + (np.flatnonzero(np.diff(frames) != 1) + 1).tolist() + [len(frames)]
    return list(zip(points[:-1], points[1:]))


def valid_window_starts(frame_index, window_length=512):
    """List start indices for windows with consecutive frames."""
    if isinstance(window_length, bool) or not isinstance(window_length, int) or window_length <= 0:
        raise ValueError("window_length must be a positive integer")
    return [
        start
        for first, end in contiguous_runs(frame_index)
        for start in range(first, end - window_length + 1)
    ]


class _WindowDataset(Dataset):
    """Validate scan arrays and crop them at the same indices."""

    def __init__(self, caches: Sequence, *, paired: bool):
        if not caches:
            raise ValueError("At least one scan cache is required")
        self.paired = paired
        self._records = []
        for cache in caches:
            names = ("bold", "frame_index", "eeg") if paired else ("bold", "frame_index")
            arrays = {}
            for name in names:
                if hasattr(cache, name):
                    arrays[name] = np.asarray(getattr(cache, name))
                else:
                    path = getattr(cache, name + "_path", None)
                    if path is None:
                        path = Path(cache.root) / f"{name}.npy"
                    arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            bold, frames = arrays["bold"], arrays["frame_index"]
            if bold.ndim != 2 or bold.shape[1] != ROI_COUNT:
                raise ValueError("BOLD must have shape [T, 64]")
            if frames.shape != (len(bold),) or not np.issubdtype(frames.dtype, np.integer):
                raise ValueError("frame_index must be a one-dimensional integer array of length T")
            if not np.isfinite(bold).all():
                raise ValueError("BOLD must contain only finite values")
            if getattr(cache, "length", len(bold)) != len(bold):
                raise ValueError("Cache length does not match the arrays")
            if paired:
                eeg = arrays["eeg"]
                if eeg.shape != (len(bold),) + EEG_SHAPE or not np.isfinite(eeg).all():
                    raise ValueError("EEG must be a finite array with shape [T, 26, 16, 200]")
            starts = tuple(valid_window_starts(frames))
            if not starts:
                raise ValueError("Scan {} has no contiguous 512-frame window".format(cache.scan_id))
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
            raise ValueError("Invalid window start")
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
    """Sample windows using the reference seed, epoch, and worker sequence."""

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
    """Sample one paired EEG and BOLD window from a scan."""

    def __init__(self, caches, *, seed=20261012):
        super().__init__(caches, seed=seed, paired=True)


class RandomFixedWindowBOLDDataset(_RandomWindowDataset):
    """Sample one BOLD window from a scan."""

    def __init__(self, caches, *, window_length=512, roi_count=64, seed=2026):
        if window_length != 512 or roi_count != 64:
            raise ValueError("The reference VAE requires window_length=512 and roi_count=64")
        self.window_length, self.roi_count = window_length, roi_count
        super().__init__(caches, seed=seed, paired=False)


class CenterPairedWindowDataset(_WindowDataset):
    """Take the center 512 frames of each contiguous scan."""

    def __init__(self, caches):
        super().__init__(caches, paired=True)
        self._starts = []
        for _, arrays, valid in self._records:
            runs = contiguous_runs(arrays["frame_index"])
            if len(runs) != 1:
                raise ValueError("Center-window evaluation requires consecutive frame indices throughout the scan")
            first, end = runs[0]
            start = first + (end - first - WINDOW_LENGTH) // 2
            if start not in valid:
                raise ValueError("The center window is shorter than 512 frames")
            self._starts.append(start)

    @property
    def starts(self):
        return tuple(zip(self.scan_ids, self._starts))

    def __getitem__(self, index):
        return self._sample(index, self._starts[index])


def paired_collate(samples):
    """Stack tensor fields and retain lists of scan identifiers."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
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
