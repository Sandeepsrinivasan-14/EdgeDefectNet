"""ResNet-18 adapted to single-channel (grayscale) 224x224 input."""

from pathlib import Path
from typing import cast

import torch
from torch import nn
from torchvision import models

from .classes import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, imagenet_weights: str | Path | None = None) -> nn.Module:
    """Build the network.

    imagenet_weights: optional path to a torchvision-compatible ResNet-18 ImageNet state_dict. Its RGB
    first-layer filters are averaged into one grayscale channel so the pretrained features carry over.
    """
    model = models.resnet18(weights=None)
    if imagenet_weights:
        state = torch.load(imagenet_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    rgb_conv = model.conv1
    gray_conv = nn.Conv2d(1, rgb_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        gray_conv.weight.copy_(rgb_conv.weight.mean(dim=1, keepdim=True))
    model.conv1 = gray_conv
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return cast(nn.Module, model)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = build_model()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
