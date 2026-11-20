"""Convert EEGLAB recordings and DiFuMo-64 ROI tables to SeqEFT caches."""

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
    """Return EEG in microvolts and the original indices of retained R149 triggers."""
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
        raise ValueError("EEG is missing the R149 fMRI trigger")
    events = mne.pick_events(events, include=event_id["R149"])
    # Load recordings that were already at 200 Hz before filtering.
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=None)
    epochs = mne.Epochs(
        raw, events, event_id={"R149": event_id["R149"]},
        tmin=-16.0, tmax=0.0, baseline=(None, None),
        reject=None, preload=True,
    )
    # Baseline correction uses all 3,201 samples before cropping off the endpoint.
    eeg = epochs.get_data(units="uV")[:, :, :3200]
    eeg = eeg.reshape(eeg.shape[0], 26, 16, 200).astype(np.float32, copy=False)
    return eeg, epochs.selection.astype(np.int64)


def _validate_bold_schema(frame):
    columns = tuple(frame.columns)
    if len(columns) != 66 or columns[-2:] != _GLOBAL_SIGNAL_COLUMNS:
        raise ValueError("BOLD must contain 66 columns: 64 ROIs followed by the two global signals")
    roi_columns = columns[:64]
    if any(not isinstance(name, str) or not name.strip() for name in roi_columns):
        raise ValueError("ROI column names must be nonempty strings")
    if len(set(roi_columns)) != 64:
        raise ValueError("ROI column names must be unique")
    return roi_columns


def normalize_bold(bold):
    """Demean [ROI, time] data and scale by the 95th percentile absolute deviation."""
    means = np.mean(bold, axis=1, keepdims=True)
    centered = bold - means
    scale = np.quantile(np.abs(centered), q=0.95, axis=1, keepdims=True) + 1e-8
    return centered / scale


def preprocess_bold(frame, selection):
    """Filter and normalize the full BOLD scan, then align it to EEG triggers."""
    from scipy.signal import butter, filtfilt

    columns = _validate_bold_schema(frame)
    bold = frame.loc[:, columns].to_numpy().T
    selection = np.asarray(selection)
    if selection.ndim != 1 or not np.issubdtype(selection.dtype, np.integer):
        raise ValueError("BOLD selection must be a one-dimensional integer array")
    if len(selection) and (selection.min() < 0 or selection.max() >= bold.shape[1]):
        raise ValueError("EEG trigger indices are outside the BOLD frame range")
    fs = 1.0 / 2.1
    b, a = butter(N=5, Wn=0.15 / (0.5 * fs), btype="low", analog=False)
    filtered = filtfilt(b, a, bold, axis=1)
    selected = normalize_bold(filtered)[:, selection].T
    return selected.astype(np.float32, copy=False), columns


def _write_scan(output_dir, scan_id, split, eeg, bold, frame_index, columns):
    """Write all four cache files before publishing the scan directory."""
    target = output_dir / scan_id
    if target.exists():
        raise FileExistsError(target)
    if eeg.shape != (len(frame_index), 26, 16, 200) or bold.shape != (len(frame_index), 64):
        raise ValueError("EEG, BOLD, and frame indices have incompatible shapes")
    if not len(frame_index) or not np.isfinite(eeg).all() or not np.isfinite(bold).all():
        raise ValueError("Preprocessed arrays must be nonempty and finite")
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
    import mne
    import pandas as pd

    eeg_dir, bold_dir, output_dir = (Path(p).expanduser().resolve()
                                     for p in (eeg_dir, bold_dir, output_dir))
    if not eeg_dir.is_dir() or not bold_dir.is_dir():
        raise FileNotFoundError("EEG or BOLD input directory does not exist")
    if output_dir.is_relative_to(eeg_dir) or output_dir.is_relative_to(bold_dir):
        raise ValueError("The output directory must be outside both source directories")
    manifest = load_split_manifest(SPLIT_FILE)
    if set(manifest) != {"train", "test"} or len(manifest["train"]) != 24 or len(manifest["test"]) != 5:
        raise ValueError("The fixed resting-state split must contain 24 training and 5 evaluation scans")
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
            raise ValueError("ROI column order differs across scans: " + scan_id)
        _write_scan(output_dir, scan_id, split, eeg, bold, selection, columns)
        print(f"{scan_id}: {len(selection)} aligned frames", flush=True)
        del raw, eeg, bold
    return {"scans": len(sources), "output_dir": str(output_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg-dir", required=True, help="Directory containing EEGLAB .set and .fdt files")
    parser.add_argument("--bold-dir", required=True, help="Directory containing DiFuMo-64 ROI tables")
    parser.add_argument("--output-dir", required=True, help="Directory for the generated scan caches")
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.eeg_dir, args.bold_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
