from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, target = self.dataset[index]
        return image, int(target), int(index)


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int]:
        source_index = int(self.indices[position])
        image, target = self.dataset[source_index]
        return image, int(target), source_index


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def _worker_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset: Dataset, batch_size: int, workers: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=_worker_init,
        generator=generator,
        drop_last=False,
    )


def load_cifar10_train(data_root: Path) -> tuple[Dataset, Dataset]:
    root = data_root / "CIFAR-10"
    augmented = datasets.CIFAR10(root=str(root), train=True, transform=train_transform(), download=False)
    clean = datasets.CIFAR10(root=str(root), train=True, transform=eval_transform(), download=False)
    return augmented, clean


def load_cifar10_test(data_root: Path) -> IndexedDataset:
    root = data_root / "CIFAR-10"
    return IndexedDataset(datasets.CIFAR10(root=str(root), train=False, transform=eval_transform(), download=False))


def load_ood(name: str, data_root: Path) -> IndexedDataset:
    if name == "CIFAR-100":
        dataset: Any = datasets.CIFAR100(
            root=str(data_root / "CIFAR-100"), train=False, transform=eval_transform(), download=False
        )
    elif name == "SVHN":
        dataset = datasets.SVHN(
            root=str(data_root / "SVHN"), split="test", transform=eval_transform(), download=False
        )
    else:
        raise ValueError(f"Unsupported read-only OOD dataset: {name}")
    return IndexedDataset(dataset)
