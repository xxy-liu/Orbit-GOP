"""The same FP32 Softplus/Dirichlet equations used in all three model heads."""
import torch
from torch.nn import functional as F

def evidence_from_logits(logits):
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return F.softplus(logits.float(), beta=1.0, threshold=20.0)

def dirichlet_parameters(evidence):
    return evidence + 1.0

def uncertainty(alpha):
    return alpha.shape[-1] / alpha.sum(-1)
