import numpy as np
import cv2
import torch
from torch.nn import functional as F
from types import SimpleNamespace
cc=SimpleNamespace(CEDL_T=5)

def transformed_batch(images: torch.Tensor, rng: np.random.RandomState) -> tuple[list[torch.Tensor], np.ndarray]:
    """Exact official repository RNG call order and image operations for one batch."""
    original = images.cpu().numpy().astype(np.float32, copy=False)
    official = [images]
    trace = np.zeros((len(original), cc.CEDL_T - 1, 2), dtype=np.float64)
    for view in range(cc.CEDL_T - 1):
        kinds = rng.choice(["rotate", "shift", "add_noise"], size=len(original))
        made = np.empty_like(original)
        for i, kind in enumerate(kinds):
            image = np.transpose(original[i], (1, 2, 0))
            if kind == "rotate":
                parameter = float(rng.uniform(-15.0, 15.0))
                h, w = image.shape[:2]
                matrix = cv2.getRotationMatrix2D((w // 2, h // 2), parameter, 1.0)
                value = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                kind_id = 0
            elif kind == "shift":
                parameter = float(rng.randint(-2, 2))
                value = np.roll(image, shift=int(parameter), axis=0)
                kind_id = 1
            else:
                parameter = 0.01
                value = (image + rng.normal(0.0, 0.01, image.shape)).astype(np.float32)
                kind_id = 2
            made[i] = np.transpose(np.clip(value, 0.0, 1.0).astype(np.float32), (2, 0, 1))
            trace[i, view] = (kind_id, parameter)
        official.append(torch.from_numpy(made))
    brightness = torch.clamp(images * 0.9, 0.0, 1.0)
    shift = F.pad(images, (1, 1, 1, 1), mode="reflect")[..., 0:32, 0:32]
    return [*official, brightness, shift], trace
