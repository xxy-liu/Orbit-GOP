from __future__ import annotations
import hashlib,json,math,random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms
K=10
MEAN=(0.4914,0.4822,0.4465)
STD=(0.2470,0.2435,0.2616)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")



def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))



def configure(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    return {
        "seed": seed,
        "deterministic_algorithms": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }



class IndexedSubset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int):
        source = int(self.indices[index])
        image, label = self.base[source]
        return image, int(label), source



class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return image, int(label), int(index)



def worker_init(_worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)



def loader(dataset: Dataset, batch: int, shuffle: bool, seed: int, workers: int = 4) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      worker_init_fn=worker_init, generator=generator)



def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])



def eval_transform(normalized: bool = True) -> transforms.Compose:
    values: list[Any] = [transforms.ToTensor()]
    if normalized:
        values.append(transforms.Normalize(MEAN, STD))
    return transforms.Compose(values)



def edl_loss(alpha: torch.Tensor, labels: torch.Tensor, epoch: int) -> dict[str, torch.Tensor]:
    with torch.autocast(device_type=alpha.device.type, enabled=False):
        alpha = alpha.float()
        onehot = F.one_hot(labels, K).float()
        strength = alpha.sum(1, keepdim=True)
        probability = alpha / strength
        error = ((onehot - probability) ** 2).sum(1).mean()
        variance = (alpha * (strength - alpha) / (strength.square() * (strength + 1))).sum(1).mean()
        masked = onehot + (1 - onehot) * alpha
        beta = torch.ones_like(masked)
        sa = masked.sum(1, keepdim=True)
        sb = beta.sum(1, keepdim=True)
        logba = torch.lgamma(masked).sum(1, keepdim=True) - torch.lgamma(sa)
        logbb = torch.lgamma(beta).sum(1, keepdim=True) - torch.lgamma(sb)
        kl = (-logba + logbb + ((masked - beta) * (torch.digamma(masked) - torch.digamma(sa))).sum(1, keepdim=True)).mean()
        anneal = min(1.0, epoch / 10.0)
        total = error + variance + anneal * kl
    return {"total": total, "error": error, "variance": variance, "kl": kl}



def learning_rate(epoch: int) -> float:
    if epoch <= 5:
        return 0.1 * epoch / 5.0
    progress = (epoch - 5) / 195.0
    return 1e-6 + 0.5 * (0.1 - 1e-6) * (1 + math.cos(math.pi * progress))



def classification_arrays(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {key: [] for key in ("sample_index", "label", "raw", "evidence", "probability", "uncertainty")}
    model.eval()
    with torch.inference_mode():
        for images, labels, indices in data_loader:
            output = model(images.to(device, non_blocking=True))
            values = {"sample_index": indices, "label": labels, "raw": output["raw_output"],
                      "evidence": output["evidence"], "probability": output["probability"],
                      "uncertainty": output["uncertainty"]}
            for key, value in values.items():
                fields[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(value) for key, value in fields.items()}



def classification_metrics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    probability = arrays["probability"].astype(np.float64)
    labels = arrays["label"].astype(np.int64)
    prediction = probability.argmax(1)
    truep = probability[np.arange(labels.size), labels]
    hist = np.bincount(prediction, minlength=K)
    return {
        "n": int(labels.size), "accuracy": float(np.mean(prediction == labels)),
        "nll": float(-np.log(np.clip(truep, 1e-12, 1)).mean()),
        "mean_total_evidence": float(arrays["evidence"].sum(1).mean()),
        "mean_uncertainty": float(arrays["uncertainty"].mean()),
        "prediction_histogram": hist.tolist(),
        "nan_inf_count": int(sum(np.size(v) - np.isfinite(v).sum() for v in arrays.values() if np.issubdtype(v.dtype, np.floating))),
        "mean_max_probability": float(probability.max(1).mean()),
    }

