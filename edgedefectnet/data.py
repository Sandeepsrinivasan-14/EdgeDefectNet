"""PyTorch datasets and loaders for <root>/{train,val,test}/<class>/*.png."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms as T

from .classes import CLASS_NAMES
from .preprocessing import IMAGE_SIZE, NORM_MEAN, NORM_STD


def get_transforms(train: bool) -> T.Compose:
    steps: list = [T.Grayscale(num_output_channels=1)]
    if train:
        steps += [
            T.RandomRotation(10),
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
        ]
    else:
        steps.append(T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True))
    steps += [T.ToTensor(), T.Normalize([NORM_MEAN], [NORM_STD])]
    return T.Compose(steps)


def image_folder(root: str | Path, split: str, train: bool = False) -> datasets.ImageFolder:
    ds = datasets.ImageFolder(Path(root) / split, transform=get_transforms(train))
    # ImageFolder sorts class folders alphabetically; this must match the model's output order.
    if ds.classes != CLASS_NAMES:
        raise ValueError(f"class folders {ds.classes} do not match CLASS_NAMES {CLASS_NAMES}")
    return ds


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def class_weights(ds: datasets.ImageFolder, max_weight: float = 10.0) -> torch.Tensor:
    """Inverse-frequency loss weights, normalised to mean 1 and clipped so tiny classes don't dominate."""
    counts = torch.bincount(torch.tensor(ds.targets), minlength=len(CLASS_NAMES)).float()
    weights = counts.sum() / (len(counts) * counts.clamp(min=1))
    return weights.clamp(max=max_weight)
