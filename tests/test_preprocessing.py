"""Check reference preprocessing using synthetic signals."""

import numpy as np
import pytest

from seqeft.prepare_data import VU_CHANNELS, _epoch_eeg, normalize_bold, preprocess_bold


def test_normalization_uses_full_scan_and_preserves_roi_amplitude_invariance():
    time = np.linspace(0, 4 * np.pi, 128)
    signal = np.stack([np.sin(time + index / 10) for index in range(64)])
    scales = np.linspace(1, 4, 64)[:, None]
    offsets = np.linspace(-20, 20, 64)[:, None]
    shifted = signal * scales + offsets
    normalized = normalize_bold(shifted)
    centered = shifted - shifted.mean(axis=1, keepdims=True)
    expected = centered / (np.quantile(np.abs(centered), 0.95, axis=1, keepdims=True) + 1e-8)
    np.testing.assert_array_equal(normalized, expected)
    np.testing.assert_allclose(normalized.mean(axis=1), 0, atol=1e-14)
    np.testing.assert_allclose(np.quantile(np.abs(normalized), 0.95, axis=1), 1, atol=2e-8)


def _bold_frame():
    pd = pytest.importorskip("pandas")
    time = np.arange(128) * 2.1
    values = np.stack([
        (index + 1) * np.sin(2 * np.pi * (0.01 + index * 0.001) * time) + index
        for index in range(64)
    ], axis=1)
    columns = [f"roi_{index:02d}" for index in range(64)]
    return pd.DataFrame(
        np.column_stack((values, np.ones((128, 2)))),
        columns=columns + ["global signal clean", "global signal raw"],
    )


def test_bold_filters_and_normalizes_before_event_selection():
    scipy_signal = pytest.importorskip("scipy.signal")
    frame = _bold_frame()
    selection = np.array([8, 19, 45, 98], dtype=np.int64)
    result, columns = preprocess_bold(frame, selection)
    raw = frame.iloc[:, :64].to_numpy().T
    b, a = scipy_signal.butter(5, 0.15 / (0.5 / 2.1), btype="low", analog=False)
    filtered = scipy_signal.filtfilt(b, a, raw, axis=1)
    centered = filtered - filtered.mean(axis=1, keepdims=True)
    expected = centered / (np.quantile(np.abs(centered), 0.95, axis=1, keepdims=True) + 1e-8)
    np.testing.assert_array_equal(result, expected[:, selection].T.astype(np.float32))
    assert tuple(columns) == tuple(frame.columns[:64])
    assert result.dtype == np.float32
    assert not np.allclose(result, normalize_bold(filtered[:, selection]).T)


@pytest.mark.parametrize("selection", [np.array([-1]), np.array([128]), np.array([1.5])])
def test_bold_rejects_invalid_event_indices(selection):
    pytest.importorskip("scipy.signal")
    with pytest.raises(ValueError):
        preprocess_bold(_bold_frame(), selection)


def test_bold_rejects_swapped_global_signal_schema():
    pytest.importorskip("scipy.signal")
    frame = _bold_frame()
    frame.columns = list(frame.columns[:-2]) + ["global signal raw", "global signal clean"]
    with pytest.raises(ValueError, match="66"):
        preprocess_bold(frame, np.array([8], dtype=np.int64))


def test_eeg_baseline_precedes_crop_and_selection_preserves_volume_indices():
    mne = pytest.importorskip("mne")
    pytest.importorskip("scipy.signal")
    sfreq = 200
    time = np.arange(5000) / sfreq
    names = [name.lower() + "'" for name in reversed(VU_CHANNELS)] + ["eog1"]
    values = np.stack([
        1e-6 * (np.sin(2 * np.pi * (1 + index / 30) * time) + index + 0.01 * time)
        for index in range(len(names))
    ])
    raw = mne.io.RawArray(values, mne.create_info(names, sfreq, ch_types="eeg"), verbose=False)
    raw.set_annotations(mne.Annotations(
        onset=[0.0, 12.0, 17.0, 19.1, 21.2],
        duration=[0.0] * 5,
        description=["R149", "noise", "R149", "R149", "R149"],
    ))
    eeg, selection = _epoch_eeg(raw)
    np.testing.assert_array_equal(selection, [1, 2, 3])
    assert tuple(raw.ch_names) == VU_CHANNELS
    assert eeg.shape == (3, 26, 16, 200)
    assert eeg.dtype == np.float32
    # Recompute baseline correction over all 3,201 samples of each filtered epoch.
    for index, onset in enumerate((17.0, 19.1, 21.2)):
        end = round(onset * sfreq)
        epoch = raw.get_data(start=end - 3200, stop=end + 1)
        expected = ((epoch - epoch.mean(axis=1, keepdims=True))[:, :3200] * 1e6).astype(np.float32)
        np.testing.assert_allclose(eeg[index].reshape(26, 3200), expected, rtol=1e-6, atol=1e-6)
