from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_alpha(alpha: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(targets, num_classes=alpha.shape[1]).to(alpha.dtype)
    return one_hot + (1.0 - one_hot) * alpha


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    beta = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    sum_beta = beta.sum(dim=1, keepdim=True)
    log_b_alpha = torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(sum_alpha)
    log_b_beta = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(sum_beta)
    expectation = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(
        dim=1, keepdim=True
    )
    return (-log_b_alpha + log_b_beta + expectation).squeeze(1)


def edl_loss(alpha: torch.Tensor, targets: torch.Tensor, epoch: int, annealing_epochs: int = 10) -> dict[str, torch.Tensor]:
    if alpha.dtype != torch.float32:
        raise TypeError("Stage B0 EDL loss requires float32 alpha")
    if epoch < 1:
        raise ValueError("epoch is one-based and must be positive")
    with torch.autocast(device_type=alpha.device.type, enabled=False):
        alpha = alpha.float()
        one_hot = F.one_hot(targets, num_classes=alpha.shape[1]).float()
        strength = alpha.sum(dim=1, keepdim=True)
        probability = alpha / strength
        error_per_sample = ((one_hot - probability) ** 2).sum(dim=1)
        variance_per_sample = (
            alpha * (strength - alpha) / (strength.square() * (strength + 1.0))
        ).sum(dim=1)
        error = error_per_sample.mean()
        variance = variance_per_sample.mean()
        bayes_risk = error + variance
        kl = dirichlet_kl_to_uniform(masked_alpha(alpha, targets)).mean()
        annealing = min(1.0, float(epoch) / float(annealing_epochs))
        total = bayes_risk + annealing * kl
    return {
        "total": total,
        "bayes_risk": bayes_risk,
        "error": error,
        "variance": variance,
        "kl": kl,
        "annealing": torch.tensor(annealing, dtype=torch.float32, device=alpha.device),
    }
