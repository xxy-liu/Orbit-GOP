"""Extracted audited GradNorm adapter. See docs/THIRD_PARTY_PROVENANCE.md and licenses/.
Original GradNorm: deeplearning-wisc/gradnorm_ood (Apache-2.0).
OpenOOD reference: Jingkang Yang, MIT License.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class EDLAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, return_feature=False):
        output = self.model(x)
        if return_feature:
            return output['raw_output'], output['features']
        return output

    def get_fc(self):
        fc = self.model.classifier
        return fc.weight.detach().cpu().numpy().copy(), fc.bias.detach().cpu().numpy().copy()


def state_hash(model):
    import hashlib
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class AuditedGradNorm:
    """OpenOOD per-image backward loop, with finite-gradient and state checks.

    Only a detached, identical copy of the final linear layer receives gradients.
    No optimizer exists. Loss and reductions match the public implementation.
    """
    def __init__(self, adapter):
        w, b = adapter.get_fc()
        self.fc = nn.Linear(*w.shape[::-1])
        self.fc.weight.data[...] = torch.from_numpy(w)
        self.fc.bias.data[...] = torch.from_numpy(b)
        self.fc.cuda()
        self.targets = torch.ones((1, 10), device='cuda')
        self.initial_state = state_hash(self.fc)

    def score_features(self, features):
        scores, maxima = [], []
        with torch.enable_grad():
            for feature in features.detach():
                self.fc.zero_grad()
                loss = torch.mean(torch.sum(
                    -self.targets * F.log_softmax(self.fc(feature[None]), dim=-1), dim=-1))
                loss.backward()
                gradient = self.fc.weight.grad
                assert torch.isfinite(gradient).all().item(), 'Nonfinite classifier gradient'
                assert torch.isfinite(self.fc.bias.grad).all().item(), 'Nonfinite bias gradient'
                assert torch.isfinite(loss).item(), 'Nonfinite loss'
                scores.append(torch.sum(torch.abs(gradient)).detach())
                maxima.append(gradient.detach().abs().max())
        raw_norm = torch.stack(scores).cpu().numpy()
        assert np.isfinite(raw_norm).all() and (raw_norm >= 0).all()
        return raw_norm, float(torch.stack(maxima).max().cpu())

    def unchanged(self):
        return state_hash(self.fc) == self.initial_state
