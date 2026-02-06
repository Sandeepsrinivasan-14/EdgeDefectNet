from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict

DATASET_ROOT = Path(r"C:\EdgeDefectNet\dataset")

splits = ["train", "val", "test"]
stats = defaultdict(lambda: {"sum": 0.0, "sum2": 0.0, "n": 0})

for split in splits:
    split_dir = DATASET_ROOT / split
    if not split_dir.exists():
        continue

    for cls_dir in split_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        for img_path in cls_dir.glob("*.png"):
            img = Image.open(img_path)
            arr = np.array(img, dtype=np.float32) / 255.0
            s = arr.sum()
            s2 = (arr ** 2).sum()
            n = arr.size
            stats[cls]["sum"] += s
            stats[cls]["sum2"] += s2
            stats[cls]["n"] += n

print("Per-class pixel mean/std (0–1):")
for cls, d in sorted(stats.items()):
    mean = d["sum"] / d["n"]
    var = d["sum2"] / d["n"] - mean ** 2
    std = float(np.sqrt(max(var, 0.0)))
    print(f"{cls:10s}: mean={mean:.4f}, std={std:.4f}")
