from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

from .config_phase1 import CLASS_NAMES, NORM_MEAN, NORM_STD, DATASET_ROOT


def get_transforms(train: bool = True):
    base = []
    if train:
        base.extend([
            T.RandomRotation(10),
            T.RandomResizedCrop(224, scale=(0.9, 1.0)),
        ])
    else:
        base.append(T.Resize((224, 224)))

    base.extend([
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[NORM_MEAN], std=[NORM_STD]),
    ])
    return T.Compose(base)


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    root = Path(DATASET_ROOT)

    train_ds = datasets.ImageFolder(root / "train", transform=get_transforms(train=True))
    val_ds   = datasets.ImageFolder(root / "val",   transform=get_transforms(train=False))
    test_ds  = datasets.ImageFolder(root / "test",  transform=get_transforms(train=False))

    print("ImageFolder classes:", train_ds.classes)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_dl, val_dl, test_dl, train_ds, _, _ = get_dataloaders(batch_size=8)
    images, labels = next(iter(train_dl))
    print("Batch images shape:", images.shape)   # expect [B, 1, 224, 224]
    print("Batch labels:", labels[:8])
