"""Test split validation, synchronized crops, and scan metrics."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from seqeft.data import (CenterPairedWindowDataset, RandomFixedWindowBOLDDataset,
                         load_split_manifest, valid_window_starts)
from seqeft.metrics import scan_metrics


def cache(tmp_path, paired=False):
    length = 520
    time = np.arange(length, dtype=np.float32)
    value = SimpleNamespace(root=tmp_path, bold=np.broadcast_to(time[:, None], (length, 64)),
                            frame_index=np.arange(length) + 7, length=length,
                            scan_id="sub01-scan01", subject_id="sub01", split="train")
    if paired:
        value.eeg = np.broadcast_to(time[:, None, None, None], (length, 26, 16, 200))
    return value


def test_split_rejects_subject_leakage(tmp_path):
    path = tmp_path / "split.yaml"
    path.write_text("train: [sub01-scan01]\ntest: [sub01-scan02]\n")
    with pytest.raises(ValueError, match="participant"):
        load_split_manifest(path)


def test_windows_do_not_cross_frame_gaps():
    frames = np.concatenate([np.arange(512), np.arange(600, 1112)])
    assert valid_window_starts(frames) == [0, 512]


def test_random_crops_are_reproducible_and_synchronized(tmp_path):
    first = RandomFixedWindowBOLDDataset([cache(tmp_path)], seed=2026)
    second = RandomFixedWindowBOLDDataset([cache(tmp_path)], seed=2026)
    for epoch in (1, 2, 3):
        first.set_epoch(epoch)
        second.set_epoch(epoch)
        for _ in range(5):
            a, b = first[0], second[0]
            assert a["start"] == b["start"]
            torch.testing.assert_close(a["bold"], b["bold"])
            assert a["bold"][0, 0] == a["start"]
            assert a["frame_index"][0] == a["start"] + 7


def test_center_eeg_and_bold_use_same_frames(tmp_path):
    dataset = CenterPairedWindowDataset([cache(tmp_path, paired=True)])
    value = dataset[0]
    assert value["start"] == 4
    assert value["eeg"].shape == (512, 26, 16, 200)
    assert value["bold"][0, 0] == value["eeg"][0, 0, 0, 0] == 4
    assert value["bold"][-1, 0] == value["eeg"][-1, 0, 0, 0] == 515
    assert value["frame_index"][0] == 11


@pytest.mark.parametrize("failure", ["short", "gap"])
def test_invalid_scan_cannot_be_silently_excluded(tmp_path, failure):
    valid = cache(tmp_path)
    invalid = cache(tmp_path)
    invalid.scan_id, invalid.subject_id = "sub02-scan01", "sub02"
    if failure == "short":
        invalid.bold = invalid.bold[:511]
        invalid.frame_index = invalid.frame_index[:511]
        invalid.length = 511
    else:
        invalid.frame_index[260:] += 10
    with pytest.raises(ValueError, match="sub02-scan01"):
        RandomFixedWindowBOLDDataset([valid, invalid])


def test_perfect_prediction_and_invalid_values():
    target = np.random.default_rng(42).normal(size=(512, 64))
    result = scan_metrics(target, target)
    for key in ("tcorr", "pixcorr", "f1_025", "f1_05"):
        assert result[key] == pytest.approx(1.0)
    assert result["bold_mse"] == result["conn_mse"] == 0
    with pytest.raises(ValueError):
        scan_metrics(np.zeros_like(target), target)
    invalid = target.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError):
        scan_metrics(invalid, target)
