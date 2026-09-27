from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve


def expected_calibration_error(probability: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(mask):
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def aurc(uncertainty: np.ndarray, correct: np.ndarray, sample_indices: np.ndarray) -> float:
    order = np.lexsort((sample_indices, uncertainty))
    errors = (~correct[order]).astype(np.float64)
    cumulative_risk = np.cumsum(errors) / np.arange(1, errors.size + 1)
    return float(cumulative_risk.mean())


def classification_metrics(arrays: dict[str, np.ndarray], ece_bins: int = 15) -> dict[str, float | int]:
    probability = arrays["probability"].astype(np.float64)
    labels = arrays["labels"].astype(np.int64)
    prediction = probability.argmax(axis=1)
    correct = prediction == labels
    uncertainty = arrays["uncertainty"].astype(np.float64)
    true_probability = probability[np.arange(labels.size), labels]
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[labels]
    error = (~correct).astype(np.int64)
    metrics: dict[str, float | int] = {
        "n": int(labels.size),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "categorical_nll": float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean()),
        "brier": float(np.square(probability - one_hot).sum(axis=1).mean()),
        "ece": expected_calibration_error(probability, labels, ece_bins),
        "aurc": aurc(uncertainty, correct, arrays["sample_index"]),
        "mean_uncertainty": float(uncertainty.mean()),
        "correct_mean_uncertainty": float(uncertainty[correct].mean()),
        "wrong_mean_uncertainty": float(uncertainty[~correct].mean()) if np.any(~correct) else float("nan"),
        "mean_total_evidence": float(arrays["evidence"].sum(axis=1).mean()),
        "error_count": int(error.sum()),
    }
    if np.unique(error).size == 2:
        metrics["error_auroc"] = float(roc_auc_score(error, uncertainty))
        metrics["error_aupr"] = float(average_precision_score(error, uncertainty))
    else:
        metrics["error_auroc"] = float("nan")
        metrics["error_aupr"] = float("nan")
    return metrics


def ood_metrics(id_uncertainty: np.ndarray, ood_uncertainty: np.ndarray) -> dict[str, float | int]:
    labels = np.concatenate([
        np.zeros(id_uncertainty.size, dtype=np.int64),
        np.ones(ood_uncertainty.size, dtype=np.int64),
    ])
    scores = np.concatenate([id_uncertainty, ood_uncertainty]).astype(np.float64)
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= 0.95)
    fpr95 = float(fpr[valid[0]]) if valid.size else 1.0
    return {
        "id_n": int(id_uncertainty.size),
        "ood_n": int(ood_uncertainty.size),
        "uncertainty_auroc": float(roc_auc_score(labels, scores)),
        "uncertainty_aupr": float(average_precision_score(labels, scores)),
        "fpr95": fpr95,
        "id_mean_uncertainty": float(id_uncertainty.mean()),
        "ood_mean_uncertainty": float(ood_uncertainty.mean()),
    }
