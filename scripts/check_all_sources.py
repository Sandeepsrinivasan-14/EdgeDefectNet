import os
from collections import Counter, defaultdict
from PIL import Image

ROOTS = {
    "IESA": r"C:\EdgeDefectNet\IESA_final_dataset_preview",
    "WM811K_IMG": r"C:\EdgeDefectNet\data\wm811k_images",
    "MULTI_CLASS": r"C:\EdgeDefectNet\data\multi_class_wafer",
    "ANOM_WAFER": r"C:\EdgeDefectNet\data\anom_wafer",
    "MIXEDTYPE": r"C:\EdgeDefectNet\data\mixedtype_wafer",
    "WAFERMAP_REPO": r"C:\EdgeDefectNet\data\WaferMap",
}

image_exts = (".png", ".jpg", ".jpeg", ".bmp")

for name, base in ROOTS.items():
    if not os.path.isdir(base):
        print(f"\n=== {name}: PATH NOT FOUND ({base}) ===")
        continue

    print(f"\n############################")
    print(f"### SOURCE: {name}")
    print(f"### PATH:   {base}")
    print(f"############################")

    mode_counter = Counter()
    size_counter = Counter()
    folder_counts = defaultdict(int)
    total_images = 0

    for root, dirs, files in os.walk(base):
        for f in files:
            if not f.lower().endswith(image_exts):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, base)
            parts = rel.split(os.sep)
            class_folder = parts[-2] if len(parts) >= 2 else "UNKNOWN"

            try:
                img = Image.open(path)
                mode_counter[img.mode] += 1
                size_counter[img.size] += 1
                folder_counts[class_folder] += 1
                total_images += 1
                img.close()
            except Exception as e:
                print("[ERROR]", path, e)

    print(f"Total images found: {total_images}")
    print("Image modes:")
    for mode, c in mode_counter.most_common():
        print(f"  {mode}: {c}")

    print("Top 5 sizes:")
    for (w, h), c in size_counter.most_common(5):
        print(f"  {(w, h)} : {c}")

    print("Top 15 folder counts:")
    for folder, c in sorted(folder_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {folder:25s} : {c}")
