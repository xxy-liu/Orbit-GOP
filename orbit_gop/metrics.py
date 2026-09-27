from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
K=10

def ood_metrics(id_score: np.ndarray, ood_score: np.ndarray) -> dict[str, float]:
    labels = np.r_[np.zeros(len(id_score), dtype=np.int8), np.ones(len(ood_score), dtype=np.int8)]
    values = np.r_[id_score, ood_score].astype(np.float64)
    fpr, tpr, _ = roc_curve(labels, values, pos_label=1, drop_intermediate=False)
    point = int(np.flatnonzero(tpr >= 0.95)[0])
    return {"AUROC": float(roc_auc_score(labels, values)), "FPR95": float(fpr[point])}



def ece(probability: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probability.max(1)
    prediction = probability.argmax(1)
    correct = prediction == labels
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for i in range(bins):
        mask = (confidence >= edges[i]) & (confidence <= edges[i + 1] if i == bins - 1 else confidence < edges[i + 1])
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)



def aurc(uncertainty: np.ndarray, correct: np.ndarray, indices: np.ndarray) -> float:
    order = np.lexsort((indices, uncertainty))
    error = (~correct[order]).astype(np.float64)
    return float((np.cumsum(error) / np.arange(1, len(error) + 1)).mean())



def id_metrics(probability: np.ndarray, uncertainty: np.ndarray, labels: np.ndarray, prediction: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    onehot = np.eye(K)[labels]
    return {
        "NLL": float(-np.log(np.clip(probability[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "Brier": float(np.square(probability - onehot).sum(1).mean()),
        "ECE15": ece(probability, labels),
        "AURC": aurc(uncertainty, prediction == labels, indices),
    }

