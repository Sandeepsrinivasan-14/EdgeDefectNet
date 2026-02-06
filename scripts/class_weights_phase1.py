from collections import Counter
from pathlib import Path

from torchvision import datasets

from .config_phase1 import IMAGEFOLDER_CLASSES, DATASET_ROOT


def get_train_counts():
    root = Path(DATASET_ROOT)
    train_ds = datasets.ImageFolder(root / "train")

    counts = Counter()
    for _, label in train_ds.samples:
        cls_name = train_ds.classes[label]
        counts[cls_name] += 1
    return counts, train_ds.classes


if __name__ == "__main__":
    counts, folder_classes = get_train_counts()

    print("ImageFolder class order:", folder_classes)
    print("Train counts per class:")
    for c in IMAGEFOLDER_CLASSES:
        print(f"  {c:10s}: {counts.get(c, 0)}")

    total = sum(counts.values())
    print("\nClass weights (inverse freq, ImageFolder order):")
    for c in IMAGEFOLDER_CLASSES:
        n = counts.get(c, 0)
        w = total / n if n > 0 else 0.0
        print(f"  {c:10s}: {w:.2f}")
