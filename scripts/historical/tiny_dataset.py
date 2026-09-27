from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class TinyManifest(Dataset):
    def __init__(self, manifest, image_root):
        self.manifest = Path(manifest)
        self.image_root = Path(image_root)
        self.rows = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            relative, label = line.rsplit(" ", 1)
            self.rows.append((relative, int(label)))
        if len(self.rows) != 7793 or len({row[0] for row in self.rows}) != len(self.rows):
            raise RuntimeError("Tiny ImageNet frozen manifest identity failure")
        tin = self.image_root / "tin"
        annotation = {}
        for line in (tin / "val" / "val_annotations.txt").read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            annotation[fields[0]] = fields[1]
        wnids = (tin / "wnids.txt").read_text(encoding="utf-8").splitlines()
        mapping = {value: index for index, value in enumerate(wnids)}
        self.true_labels = np.asarray([mapping[annotation[Path(row[0]).name]] for row in self.rows], dtype=np.int64)
        self.transform = transforms.Compose([
            transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        relative, _ = self.rows[index]
        with Image.open(self.image_root / relative) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.true_labels[index]), int(index)
