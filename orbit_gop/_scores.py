from __future__ import annotations
import math
from typing import Any
import numpy as np
from scipy.optimize import minimize_scalar
K=10
BETA=0.40

def fit_c_global(evidence: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    evidence = evidence.astype(np.float64)
    labels = labels.astype(np.int64)
    def objective(log_c: float) -> float:
        alpha = 1.0 + math.exp(log_c) * evidence
        probability = alpha / alpha.sum(1, keepdims=True)
        return float(-np.log(np.clip(probability[np.arange(labels.size), labels], 1e-15, 1)).mean())
    fit = minimize_scalar(objective, bounds=(-8.0, 8.0), method="bounded", options={"xatol": 1e-12, "maxiter": 1000})
    return {"c_global": float(math.exp(fit.x)), "nll": float(fit.fun), "success": bool(fit.success), "nfev": int(fit.nfev), "message": str(fit.message)}



def scores(data: dict[str, np.ndarray], q: np.ndarray, c_global: float) -> dict[str, np.ndarray]:
    evidence = data["evidence"].astype(np.float64)
    global_alpha = 1.0 + c_global * evidence
    orbit_alpha = 1.0 + c_global * np.exp(-BETA * q)[:, None] * evidence
    return {
        "q_func": q.astype(np.float64),
        "global_uncertainty": K / global_alpha.sum(1),
        "orbit_uncertainty": K / orbit_alpha.sum(1),
        "global_probability": global_alpha / global_alpha.sum(1, keepdims=True),
        "orbit_probability": orbit_alpha / orbit_alpha.sum(1, keepdims=True),
        "pred_alpha_numeric": orbit_alpha.argmax(1).astype(np.int64),
        "orbit_alpha": orbit_alpha,
    }

