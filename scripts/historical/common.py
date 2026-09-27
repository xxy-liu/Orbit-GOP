import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.registry import create_model
from training.b0.data import IndexedDataset, IndexedSubset

ROOT = Path(__file__).resolve().parents[2]

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def verify_splits():
    expected=json.loads((ROOT/'splits/historical/split_hashes.json').read_text())
    for name,digest in expected.items():
        if sha256(ROOT/'splits/historical'/name)!=digest:
            raise ValueError('Split hash mismatch: '+name)

def load_model(backbone, checkpoint, device):
    model=create_model(backbone)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(payload['model_state'],strict=True)
    model.to(device).float().eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model

def configure():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)

def dataset(name,data_root,calibration=False):
    root=Path(data_root)
    if calibration:
        verify_splits()
        base=datasets.CIFAR10(str(root/'CIFAR-10'),train=True,download=False,transform=transforms.ToTensor())
        indices=np.load(ROOT/'splits/historical/cifar10_calibration_indices.npy',allow_pickle=False)
        return IndexedSubset(base,indices)
    if name=='cifar10':base=datasets.CIFAR10(str(root/'CIFAR-10'),train=False,download=False,transform=transforms.ToTensor())
    elif name=='cifar100':base=datasets.CIFAR100(str(root/'CIFAR-100'),train=False,download=False,transform=transforms.ToTensor())
    elif name=='svhn':base=datasets.SVHN(str(root/'SVHN'),split='test',download=False,transform=transforms.ToTensor())
    else:raise ValueError('Unsupported dataset')
    return IndexedDataset(base)

def save_new(path,**values):
    path=Path(path)
    if path.suffix!='.npz':raise ValueError('Output must end in .npz')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:np.savez_compressed(stream,**values)
