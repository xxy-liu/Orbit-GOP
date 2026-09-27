from __future__ import annotations
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from .functional_risk import generalized_js
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)

def stable_margin(raw: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    mask = F.one_hot(classes, raw.shape[1]).bool()
    competitors = raw.masked_fill(mask, -torch.inf)
    return raw.gather(1, classes[:, None]).squeeze(1) - torch.logsumexp(competitors, dim=1)



def image_view(images: torch.Tensor, name: str) -> torch.Tensor:
    if name == "original":
        return images
    if name == "brightness_090":
        return torch.clamp(images * 0.90, 0.0, 1.0)
    if name == "reflect_shift_1px":
        return F.pad(images, (1, 1, 1, 1), mode="reflect")[..., :32, :32]
    raise KeyError(name)



def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)
    return (images - mean) / std



def loader(dataset: Dataset, batch: int, shuffle: bool, seed: int, workers: int = 4) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      worker_init_fn=worker_init, generator=generator)



def worker_init(_worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)



def extract_gop(model: torch.nn.Module, dataset: Dataset, device: torch.device, layers: tuple[str, ...], seed: int, batch: int = 32) -> dict[str, np.ndarray]:
    output_lists: dict[str, list[np.ndarray]] = {key: [] for key in ("sample_index", "label", "prediction", "raw", "evidence")}
    for layer in layers:
        output_lists[f"d_gop_{layer}"] = []
    model.eval()
    for images, labels, indices in loader(dataset, batch, False, seed, workers=2):
        images = images.to(device, non_blocking=True)
        fixed_classes = None
        per_layer: dict[str, list[torch.Tensor]] = {layer: [] for layer in layers}
        original = None
        for view_name in ("original", "brightness_090", "reflect_shift_1px"):
            x = normalize(image_view(images, view_name)).contiguous().float().detach().requires_grad_(True)
            current = model(x)
            if fixed_classes is None:
                fixed_classes = current["probability"].argmax(1).detach()
                original = current
            margin = stable_margin(current["raw_output"], fixed_classes)
            activations = tuple(current[layer] for layer in layers)
            gradients = torch.autograd.grad(margin.sum(), activations, retain_graph=False, create_graph=False)
            for layer, gradient in zip(layers, gradients):
                energy = gradient.square().mean((-1, -2))
                signature = (energy + 1e-12) / (energy.sum(1, keepdim=True) + energy.shape[1] * 1e-12)
                per_layer[layer].append(signature.detach())
            del current, x
        assert original is not None and fixed_classes is not None
        values = {"sample_index": indices.numpy(), "label": labels.numpy(), "prediction": fixed_classes.cpu().numpy(),
                  "raw": original["raw_output"].detach().cpu().numpy(), "evidence": original["evidence"].detach().cpu().numpy()}
        for key, value in values.items():
            output_lists[key].append(value)
        for layer in layers:
            stack = torch.stack(per_layer[layer], 1)
            output_lists[f"d_gop_{layer}"].append(generalized_js(stack).cpu().numpy())
        del original
    result = {key: np.concatenate(value) for key, value in output_lists.items()}
    for key, value in result.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise FloatingPointError(f"non-finite extraction field {key}")
    return result

