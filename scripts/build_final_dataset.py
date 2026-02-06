import os
import random
from collections import Counter
from pathlib import Path

from PIL import Image

# ====== CONFIG ======
# Sources
IESA_ROOT = Path(r"C:\EdgeDefectNet\IESA_final_dataset_preview\IESA_unique")
WM811K_ROOT = Path(r"C:\EdgeDefectNet\data\wm811k_images")
MULTI_CLASS_ROOT = Path(r"C:\EdgeDefectNet\data\multi_class_wafer")

# Output
DATASET_ROOT = Path(r"C:\EdgeDefectNet\dataset")
IMG_SIZE = (224, 224)
RANDOM_SEED = 42

# Final 9 classes
FINAL_CLASSES = [
    "Clean",
    "Shorts",
    "Opens",
    "Bridges",
    "Vias",
    "Scratches",
    "Cracks",
    "LER",
    "Other",
]

random.seed(RANDOM_SEED)

# ====== UTILITIES ======
def collect_images(root, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return paths


def save_gray_resized(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path).convert("L")
    img = img.resize(IMG_SIZE)
    img.save(dst_path)


# ====== STEP 1: COLLECT AND MAP IMAGES TO FINAL CLASSES ======
all_items = []  # list of (src_path, final_class, source_name)

# ---- 1A) IESA (Cracks, Vias, Clean) ----
if IESA_ROOT.exists():
    for p in collect_images(IESA_ROOT):
        parts = p.relative_to(IESA_ROOT).parts
        # parts: ["Cracks", "Defect", "file.png"] or ["Vias", "Clean", "file.png"]
        # map based on top-level folder
        top = parts[0].lower()
        if top == "cracks":
            final_cls = "Cracks"
        elif top == "vias":
            # Vias/Clean -> Clean; Vias/Defect -> Vias
            if len(parts) >= 2 and parts[1].lower() == "clean":
                final_cls = "Clean"
            else:
                final_cls = "Vias"
        else:
            final_cls = "Other"
        all_items.append((p, final_cls, "IESA"))

# ---- 1B) WM811K image subset (32x32, 9 wafer patterns) ----
# Known classes from Junayed’s WM811k subset[web:79]
wm_map = {
    "center": "Shorts",
    "donut": "Cracks",
    "edge local": "LER",
    "edge ring": "LER",
    "local": "Opens",
    "near full": "Bridges",
    "none": "Clean",
    "random": "Other",
    "scratch": "Scratches",
}

if WM811K_ROOT.exists():
    for p in collect_images(WM811K_ROOT):
        # folder just above the file is the pattern name
        cls_folder = p.parent.name.lower()
        cls_folder_norm = cls_folder.replace("-", " ").strip()
        final_cls = wm_map.get(cls_folder_norm, "Other")
        all_items.append((p, final_cls, "WM811K_IMG"))

# ---- 1C) Multi-class wafer dataset (640x640, same 9 patterns) ----
# Folders: Center, Donut, Edge-Loc, Edge-Ring, Local, Near-Full, Normal, Random, Scratch[web:29][web:52]
multi_map = {
    "center": "Shorts",
    "donut": "Cracks",
    "edge-loc": "LER",
    "edge-loc.": "LER",
    "edge-loc ": "LER",
    "edge-loc ": "LER",
    "edge ring": "LER",
    "edge-ring": "LER",
    "local": "Opens",
    "near-full": "Bridges",
    "near full": "Bridges",
    "normal": "Clean",
    "none": "Clean",
    "random": "Other",
    "scratch": "Scratches",
}

if MULTI_CLASS_ROOT.exists():
    for p in collect_images(MULTI_CLASS_ROOT):
        cls_folder = p.parent.name.lower()
        cls_folder_norm = cls_folder.replace("_", "-").replace(" ", "-").strip()
        # try exact, then some fallbacks
        final_cls = multi_map.get(cls_folder_norm)
        if final_cls is None:
            # try some loose normalizations
            lf = cls_folder.replace(" ", "").replace("-", "")
            if lf == "center":
                final_cls = "Shorts"
            elif lf in ("donut", "donut."):
                final_cls = "Cracks"
            elif "edge" in lf:
                final_cls = "LER"
            elif "local" in lf or lf == "loc":
                final_cls = "Opens"
            elif "nearfull" in lf:
                final_cls = "Bridges"
            elif "normal" in lf or lf == "none":
                final_cls = "Clean"
            elif "scratch" in lf:
                final_cls = "Scratches"
            elif "random" in lf:
                final_cls = "Other"
            else:
                final_cls = "Other"
        all_items.append((p, final_cls, "MULTI_CLASS"))

# Check what we collected
print("Total collected images before filtering:", len(all_items))
per_class = Counter(cls for _, cls, _ in all_items)
print("Counts per final class (raw):")
for c in FINAL_CLASSES:
    print(f"  {c:10s}: {per_class.get(c, 0)}")

# ====== STEP 2: BALANCE CLASSES & SPLIT TRAIN/VAL/TEST ======
# Simple strategy: cap each class at N images (downsample large ones)
MAX_PER_CLASS = 1500

class_to_items = {c: [] for c in FINAL_CLASSES}
for path, cls, src in all_items:
    if cls not in class_to_items:
        continue
    class_to_items[cls].append((path, src))

# Downsample
balanced_items = []
for cls, items in class_to_items.items():
    if not items:
        continue
    if len(items) > MAX_PER_CLASS:
        items = random.sample(items, MAX_PER_CLASS)
    balanced_items.extend((p, cls, src) for p, src in items)

print("\nAfter balancing (cap per class = {}):".format(MAX_PER_CLASS))
per_class_bal = Counter(cls for _, cls, _ in balanced_items)
for c in FINAL_CLASSES:
    print(f"  {c:10s}: {per_class_bal.get(c, 0)}")

# Split: 70% train, 15% val, 15% test
splits = {"train": [], "val": [], "test": []}
for cls in FINAL_CLASSES:
    items = [(p, c, s) for (p, c, s) in balanced_items if c == cls]
    if not items:
        continue
    random.shuffle(items)
    n = len(items)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    splits["train"].extend(items[:n_train])
    splits["val"].extend(items[n_train:n_train + n_val])
    splits["test"].extend(items[n_train + n_val:])

print("\nSplit sizes:")
for split_name, items in splits.items():
    print(f"  {split_name:5s}: {len(items)} images")

# ====== STEP 3: WRITE IMAGES TO OUTPUT FOLDERS ======
for split_name, items in splits.items():
    for src_path, cls, src_name in items:
        # e.g., dataset/train/Clean/WM811K_IMG_xxxx.png
        dst_dir = DATASET_ROOT / split_name / cls
        # include source name in filename to track provenance
        dst_name = f"{src_name}__{src_path.stem}.png"
        dst_path = dst_dir / dst_name
        save_gray_resized(src_path, dst_path)

print("\nDone. Final dataset created under:", DATASET_ROOT)
for split_name in ["train", "val", "test"]:
    for cls in FINAL_CLASSES:
        cls_dir = DATASET_ROOT / split_name / cls
        if cls_dir.exists():
            count = len(list(cls_dir.glob("*.png")))
            if count > 0:
                print(f"{split_name}/{cls}: {count} images")
