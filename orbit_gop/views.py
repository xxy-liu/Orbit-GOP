from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)

def make_views(images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    original = images
    brightness = torch.clamp(images * 0.9, 0.0, 1.0)
    shifted = F.pad(images, (1, 1, 1, 1), mode="reflect")[..., 0:32, 0:32]
    return original, brightness, shifted



def normalize(images: Tensor) -> Tensor:
    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)
    return (images - mean) / std

