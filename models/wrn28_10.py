from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.drop_rate = float(drop_rate)
        self.equal_in_out = in_planes == out_planes
        self.shortcut = None if self.equal_in_out else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.equal_in_out:
            out = self.relu1(self.bn1(x))
            residual = x
        else:
            x = self.relu1(self.bn1(x))
            out = x
            residual = self.shortcut(x)
        out = self.relu2(self.bn2(self.conv1(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return residual + self.conv2(out)


class NetworkBlock(nn.Module):
    def __init__(self, count: int, in_planes: int, out_planes: int, stride: int):
        super().__init__()
        self.layer = nn.Sequential(*[
            BasicBlock(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1)
            for i in range(count)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class WideResNet28x10EDL(nn.Module):
    """OpenOOD-compatible WRN-28-10 body with the frozen Softplus EDL head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        if num_classes != 10:
            raise ValueError("P6-C is frozen to CIFAR-10 K=10")
        n = 4
        channels = (16, 160, 320, 640)
        self.conv1 = nn.Conv2d(3, channels[0], 3, padding=1, bias=False)
        self.block1 = NetworkBlock(n, channels[0], channels[1], 1)
        self.block2 = NetworkBlock(n, channels[1], channels[2], 2)
        self.block3 = NetworkBlock(n, channels[2], channels[3], 2)
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(channels[3], num_classes)
        self.num_classes = num_classes
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0.0, math.sqrt(2.0 / fan))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.conv1(x)
        l1 = self.block1(x)
        l2 = self.block2(l1)
        l3 = self.block3(l2)
        post = self.relu(self.bn1(l3))
        features = F.avg_pool2d(post, 8).flatten(1)
        raw = self.classifier(features)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            raw32 = raw.float()
            evidence = F.softplus(raw32, beta=1.0, threshold=20.0)
            alpha = evidence + 1.0
            strength = alpha.sum(1)
            probability = alpha / strength[:, None]
            uncertainty = self.num_classes / strength
        return {
            "raw_output": raw32,
            "evidence": evidence,
            "alpha": alpha,
            "strength": strength,
            "probability": probability,
            "uncertainty": uncertainty,
            "features": features.float(),
            "L1": l1,
            "L2": l2,
            "L3": l3,
        }

