"""Portable FP32 adapter for the frozen three-view functional extraction.

Numerical operations follow the archived WRN extractor and VGG FP32 closure.
This adapter does not claim to reproduce historical mixed-precision scores.
"""
import numpy as np
import torch
from .views import make_views, normalize
from ._resnet_signature import stable_margin
from .functional_risk import generalized_js

LAYERS = {'resnet18': 'layer3', 'vgg16_bn': 'block4', 'wrn28_10': 'block3'}

def signature_from_gradient(gradient):
    energy = gradient.square().mean((-1, -2))
    return (energy + 1e-12) / (energy.sum(1, keepdim=True) + energy.shape[1] * 1e-12)

def extract_batch(model, images, backbone):
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError('Expected raw [N,3,32,32] images')
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError('Images must be finite and in [0,1], before normalization')
    model.eval()
    fixed = None
    signatures = []
    with torch.enable_grad(), torch.autocast(device_type=images.device.type, enabled=False):
        for view in make_views(images.float()):
            captured = {}
            handle = getattr(model, LAYERS[backbone]).register_forward_hook(
                lambda _m, _i, value: captured.__setitem__('feature', value))
            try:
                output = model(normalize(view).detach().requires_grad_(True))
            finally:
                handle.remove()
            if fixed is None:
                fixed = output['probability'].argmax(1).detach()
                evidence = output['evidence'].detach()
                raw = output['raw_output'].detach()
            gradient = torch.autograd.grad(stable_margin(output['raw_output'], fixed).sum(), captured['feature'])[0]
            signatures.append(signature_from_gradient(gradient).detach())
    stack = torch.stack(signatures, 1)
    result = {'prediction': fixed, 'evidence': evidence, 'raw': raw,
              'd_gop_L3': generalized_js(stack), 'signatures': stack}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Non-finite extraction output')
    return result

def extract_loader(model, loader, device, backbone):
    chunks = {}
    for images, labels, indices in loader:
        values = extract_batch(model, images.to(device), backbone)
        values.pop('signatures')
        values.update(label=labels, sample_index=indices)
        for key, value in values.items():
            chunks.setdefault(key, []).append(value.detach().cpu().numpy())
    if not chunks:
        raise ValueError('Empty dataset')
    return {k: np.concatenate(v) for k, v in chunks.items()}
