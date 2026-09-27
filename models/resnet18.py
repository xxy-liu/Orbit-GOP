from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class ResNet18EDLSoftplus(nn.Module):
    """CIFAR-adapted ResNet18 with a fixed Softplus evidential head."""

    def __init__(self, num_classes: int = 10, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        if num_classes != 10:
            raise ValueError("Stage B0 is registered for exactly 10 CIFAR-10 classes")
        backbone = resnet18(weights=None, num_classes=num_classes)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.classifier = backbone.fc
        self.num_classes = num_classes
        self.beta = float(beta)
        self.threshold = float(threshold)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        raw_output = self.classifier(features)
        # EDL quantities are deliberately computed in float32 even when the
        # convolutional path runs under autocast.
        with torch.autocast(device_type=raw_output.device.type, enabled=False):
            raw32 = raw_output.float()
            evidence = F.softplus(raw32, beta=self.beta, threshold=self.threshold)
            alpha = evidence + 1.0
            strength = alpha.sum(dim=1, keepdim=True)
            probability = alpha / strength
            uncertainty = self.num_classes / strength.squeeze(1)
        return {
            "raw_output": raw32,
            "evidence": evidence,
            "alpha": alpha,
            "strength": strength.squeeze(1),
            "probability": probability,
            "uncertainty": uncertainty,
            "features": features.float(),
        }
