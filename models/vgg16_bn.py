from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _block(in_channels: int, out_channels: int, convolutions: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(convolutions):
        layers.extend([
            nn.Conv2d(in_channels if index == 0 else out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
    return nn.Sequential(*layers)


class VGG16BNEDL(nn.Module):
    """Exact P5-3 CIFAR-compatible VGG16-BN EDL architecture."""

    def __init__(self, num_classes: int = 10, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        if num_classes != 10:
            raise ValueError("P5-3R is frozen for exactly 10 CIFAR-10 classes")
        self.block1 = _block(3, 64, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = _block(64, 128, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = _block(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.block4 = _block(256, 512, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.block5 = _block(512, 512, 3)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        self.num_classes = int(num_classes)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.pool5(self.block5(x))
        features = torch.flatten(self.avgpool(x), 1)
        raw_output = self.classifier(features)
        with torch.autocast(device_type=raw_output.device.type, enabled=False):
            raw32 = raw_output.float()
            evidence = F.softplus(raw32, beta=self.beta, threshold=self.threshold)
            alpha = evidence + 1.0
            strength = alpha.sum(dim=1)
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
        }

