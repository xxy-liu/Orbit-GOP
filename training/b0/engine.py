from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from .loss import edl_loss
from .metrics import classification_metrics


@torch.inference_mode()
def infer(model, loader, device: torch.device, epoch: int, annealing_epochs: int, ece_bins: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    loss_sums = {name: 0.0 for name in ("total", "bayes_risk", "error", "variance", "kl")}
    seen = 0
    amp_enabled = device.type == "cuda"
    for images, targets, sample_indices in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(images)
        losses = edl_loss(output["alpha"], targets, epoch, annealing_epochs)
        batch = int(targets.size(0))
        seen += batch
        for name in loss_sums:
            loss_sums[name] += float(losses[name].item()) * batch
        chunks["sample_index"].append(sample_indices.numpy().astype(np.int64, copy=False))
        chunks["labels"].append(targets.cpu().numpy().astype(np.int64, copy=False))
        for name in ("raw_output", "evidence", "alpha", "probability", "strength", "uncertainty", "features"):
            chunks[name].append(output[name].detach().cpu().numpy().astype(np.float32, copy=False))
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    order = np.argsort(arrays["sample_index"], kind="stable")
    arrays = {name: value[order] for name, value in arrays.items()}
    metrics = classification_metrics(arrays, ece_bins)
    metrics.update({f"loss_{name}": value / seen for name, value in loss_sums.items()})
    return metrics, arrays
