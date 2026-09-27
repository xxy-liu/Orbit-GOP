from __future__ import annotations
import numpy as np

def stable_softmax_scores(raw: np.ndarray) -> dict[str, np.ndarray]:
    raw64 = np.asarray(raw, dtype=np.float64)
    shifted = raw64 - raw64.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_p = shifted - log_norm
    probability = np.exp(log_p)
    return {
        "msp": probability.max(axis=1),
        "predictive_entropy": (probability * log_p).sum(axis=1),
        "energy_t1": raw64.max(axis=1) + np.log(np.exp(raw64 - raw64.max(axis=1, keepdims=True)).sum(axis=1)),
    }



def react_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, threshold: float) -> np.ndarray:
    clipped = np.minimum(np.asarray(features, dtype=np.float64), float(threshold))
    return clipped @ np.asarray(weight, dtype=np.float64).T + np.asarray(bias, dtype=np.float64)

