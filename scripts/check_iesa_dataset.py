import os
from collections import Counter, defaultdict
from PIL import Image

# Base directory of extracted dataset
BASE = r"C:\EdgeDefectNet\IESA_final_dataset_preview"

image_exts = (".png", ".jpg", ".jpeg", ".bmp")
mode_counter = Counter()
size_counter = Counter()
folder_counts = defaultdict(int)

for root, dirs, files in os.walk(BASE):
    for f in files:
        if not f.lower().endswith(image_exts):
            continue
        path = os.path.join(root, f)
        rel = os.path.relpath(path, BASE)
        parts = rel.split(os.sep)

        # Example: IESA_unique\Cracks\Defect\defect_0001.png
        # We treat the folder just above the image as the "class-like" folder
        class_folder = parts[-2] if len(parts) >= 2 else "UNKNOWN"

        try:
            img = Image.open(path)
            mode_counter[img.mode] += 1
            size_counter[img.size] += 1
            folder_counts[class_folder] += 1
            img.close()
        except Exception as e:
            print("[ERROR]", path, e)

print("\n=== IMAGE MODES (grayscale vs RGB) ===")
for mode, c in mode_counter.most_common():
    print(mode, ":", c)

print("\n=== TOP 10 IMAGE SIZES (W,H) ===")
for (w, h), c in size_counter.most_common(10):
    print(f"{(w,h)} : {c}")

print("\n=== COUNTS PER FOLDER (treat these as potential classes) ===")
for folder, c in sorted(folder_counts.items()):
    print(f"{folder:20s} : {c}")
