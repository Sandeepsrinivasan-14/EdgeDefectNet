from pathlib import Path
from collections import Counter, defaultdict

DATASET_ROOT = Path(r"C:\EdgeDefectNet\dataset")

splits = ["train", "val", "test"]
sources = ["IESA", "WM811K_IMG", "MULTI_CLASS"]

for split in splits:
    split_dir = DATASET_ROOT / split
    if not split_dir.exists():
        continue

    print(f"\n===== SPLIT: {split.upper()} =====")
    per_class = Counter()
    per_class_source = defaultdict(Counter)

    for cls_dir in split_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for img_path in cls_dir.glob("*.png"):
            per_class[cls_name] += 1
            # filenames are like: SOURCE__originalname.png
            src = img_path.stem.split("__", 1)[0]
            per_class_source[cls_name][src] += 1

    print("Class counts:")
    for cls, n in sorted(per_class.items()):
        print(f"  {cls:10s}: {n}")

    print("\nClass x Source:")
    for cls, src_counts in sorted(per_class_source.items()):
        parts = ", ".join(f"{s}={c}" for s, c in src_counts.items())
        print(f"  {cls:10s}: {parts}")
