from __future__ import annotations
from typing import Any
import numpy as np
from scipy.optimize import minimize_scalar
K=10

def categorical_nll(probability: np.ndarray, labels: np.ndarray) -> float:
    true_probability = probability[np.arange(labels.size), labels]
    return float(-np.log(np.clip(true_probability, 1e-15, 1.0)).mean())



def fit_global_scale(evidence: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    evidence = np.asarray(evidence, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    def objective(log_scale: float) -> float:
        scale = float(np.exp(log_scale))
        alpha = 1.0 + scale * evidence
        probability = alpha / alpha.sum(axis=1, keepdims=True)
        return categorical_nll(probability, labels)

    bounds = (-8.0, 8.0)
    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=bounds,
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    scale = float(np.exp(result.x))
    boundary_distance = float(min(result.x - bounds[0], bounds[1] - result.x))
    return {
        "c_global": scale,
        "log_c_global": float(result.x),
        "nll": float(result.fun),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "boundary_distance": boundary_distance,
        "pass": bool(result.success and np.isfinite(scale) and scale > 0 and boundary_distance > 1e-6),
    }



def scaled_probability(evidence: np.ndarray, scale: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    evidence = np.asarray(evidence, dtype=np.float64)
    factor = np.asarray(scale, dtype=np.float64)
    scaled = evidence * float(factor) if factor.ndim == 0 else evidence * factor[:, None]
    alpha = 1.0 + scaled
    probability = alpha / alpha.sum(axis=1, keepdims=True)
    uncertainty = K / alpha.sum(axis=1)
    return probability, uncertainty

