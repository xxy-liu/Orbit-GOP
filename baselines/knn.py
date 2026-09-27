from __future__ import annotations
import numpy as np
import torch
K=50
def emit(event,**payload):
    print(event, payload, flush=True)


def l2_normalize(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or float(norm.min()) <= 0.0:
        raise FloatingPointError("invalid feature norm")
    return np.ascontiguousarray(value / norm, dtype=np.float32)


def kth_distance_formula(bank: np.ndarray, query: np.ndarray, query_chunk: int = 256) -> np.ndarray:
    device = torch.device("cuda")
    bank_tensor = torch.from_numpy(bank).to(device=device, dtype=torch.float32)
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, query.shape[0], query_chunk):
            q = torch.from_numpy(query[start:start + query_chunk]).to(device=device, dtype=torch.float32)
            squared = (2.0 - 2.0 * (q @ bank_tensor.T)).clamp_min_(0.0)
            kth_squared = torch.topk(squared, k=K, dim=1, largest=False, sorted=True).values[:, -1]
            result.append(torch.sqrt(kth_squared).cpu().numpy().astype(np.float32))
            if (start // query_chunk + 1) % 20 == 0:
                emit("knn_score_progress", rows=min(start + query_chunk, query.shape[0]), total=int(query.shape[0]))
    del bank_tensor
    torch.cuda.empty_cache()
    return np.concatenate(result)
