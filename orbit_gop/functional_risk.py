from __future__ import annotations
import math
import numpy as np
import torch
K=10

def generalized_js(signatures: torch.Tensor) -> torch.Tensor:
    safe = signatures.clamp_min(torch.finfo(signatures.dtype).tiny)
    mean = safe.mean(1)
    return (-(mean * mean.log()).sum(-1) + (safe * safe.log()).sum(-1).mean(1)) / math.log(signatures.shape[1])



def midrank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=np.float64), kind="mergesort")
    values = np.asarray(values, dtype=np.float64)
    left = np.searchsorted(ref, values, side="left")
    right = np.searchsorted(ref, values, side="right")
    return (left + 0.5 * (right - left)) / ref.size



def q_func(calibration: dict[str, np.ndarray], target: dict[str, np.ndarray], layer: str) -> np.ndarray:
    result = np.empty(target["prediction"].size, dtype=np.float64)
    for class_id in range(K):
        ref = calibration[f"d_gop_{layer}"][calibration["prediction"] == class_id]
        mask = target["prediction"] == class_id
        if ref.size == 0:
            raise RuntimeError(f"empty calibration prediction class {class_id}")
        result[mask] = midrank(ref, target[f"d_gop_{layer}"][mask])
    return result

