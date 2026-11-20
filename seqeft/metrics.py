"""Scan-level BOLD and functional connectivity metrics."""
from collections import defaultdict
import math

import numpy as np

METRIC_KEYS = ("tcorr", "bold_mse", "pixcorr", "conn_mse", "f1_025", "f1_05")


def _check_constant_columns(value, name):
    centered = value - value.mean(axis=0, keepdims=True)
    if np.any(np.sum(centered * centered, axis=0) <= 0.0):
        raise ValueError(name + " contains a constant ROI time series")


def _pearson(first, second):
    value = float(np.corrcoef(first.squeeze(), second.squeeze())[0, 1])
    if not math.isfinite(value):
        raise ValueError("Pearson correlation is not finite")
    return value


def scan_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != (512, 64) or target.shape != pred.shape:
        raise ValueError("Prediction and target must both have shape [512, 64]")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Prediction and target must contain only finite values")
    _check_constant_columns(pred, "prediction")
    _check_constant_columns(target, "target")
    pred_fc = np.corrcoef(pred.T)
    target_fc = np.corrcoef(target.T)
    if not np.isfinite(pred_fc).all() or not np.isfinite(target_fc).all():
        raise ValueError("The functional connectivity matrix contains nonfinite values")
    upper = np.triu_indices(64, k=1)
    pred_edges, target_edges = pred_fc[upper], target_fc[upper]
    if np.ptp(pred_edges) <= 0.0 or np.ptp(target_edges) <= 0.0:
        raise ValueError("FC metrics require nonconstant edge vectors")

    def edge_f1(fraction):
        pred_mask = pred_edges >= np.percentile(pred_edges, 100.0 * (1.0 - fraction))
        target_mask = target_edges >= np.percentile(target_edges, 100.0 * (1.0 - fraction))
        true_positive = int(np.sum(pred_mask & target_mask))
        false_positive = int(np.sum(pred_mask & ~target_mask))
        false_negative = int(np.sum(~pred_mask & target_mask))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0

    metrics = {
        "tcorr": float(np.mean([_pearson(pred[:, i], target[:, i]) for i in range(64)])),
        "pixcorr": _pearson(pred_edges, target_edges),
        "conn_mse": float(np.mean((pred_edges - target_edges) ** 2)),
        "f1_025": edge_f1(0.25),
        "f1_05": edge_f1(0.50),
        "bold_mse": float(np.mean((pred - target) ** 2)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Scan metrics must be finite")
    return metrics


def aggregate_metrics(rows):
    """Average within participants, then compute the participant mean and sample SD."""
    if not rows:
        raise ValueError("At least one scan is required")
    subjects = defaultdict(list)
    for row in rows:
        subjects[str(row["subject_id"])].append(row)
    per_subject = {
        subject: {key: float(np.mean([row[key] for row in scans])) for key in METRIC_KEYS}
        for subject, scans in subjects.items()
    }
    return {
        "n_scans": len(rows),
        "n_subjects": len(per_subject),
        "per_subject": per_subject,
        "subject_macro": {
            key: float(np.mean([row[key] for row in per_subject.values()]))
            for key in METRIC_KEYS
        },
        "subject_sample_std": {
            key: float(np.std([row[key] for row in per_subject.values()], ddof=1))
            if len(per_subject) > 1 else None
            for key in METRIC_KEYS
        },
    }
