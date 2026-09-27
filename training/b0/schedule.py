from __future__ import annotations

import math


def learning_rate_for_epoch(epoch: int, epochs: int, lr_max: float, warmup_epochs: int, lr_min: float) -> float:
    if not 1 <= epoch <= epochs:
        raise ValueError("epoch outside configured range")
    if epoch <= warmup_epochs:
        return lr_max * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
