"""Build the EdgeDefectNet dataset with a leakage-free, group-aware train/val/test split.

Three ways to run it:

  # 1. Reproduce the exact published split from the raw sources (recommended)
  python scripts/build_dataset.py --manifest data/manifest.csv \
      --iesa raw/IESA_unique --wm811k raw/wm811k_images --multiclass raw/multi_class_wafer --out dataset

  # 2. Build a fresh split from the raw sources (writes a new manifest)
  python scripts/build_dataset.py --iesa ... --wm811k ... --multiclass ... --out dataset --write-manifest my_manifest.csv

  # 3. Re-split an already processed pool of images (folders of <split>/<class>/*.png)
  python scripts/build_dataset.py --from-processed Phase1_Dataset --out dataset --write-manifest data/manifest.csv

Output layout: <out>/{train,val,test}/<class>/<SOURCE>__<original>.png (224x224, 8-bit grayscale).
"""

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgedefectnet.classes import CLASS_NAMES
from edgedefectnet.splits import SPLITS, Item, assign_splits, group_id, read_manifest, write_manifest

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
IMG_SIZE = (224, 224)
MAX_PER_CLASS = 1500

WM_PATTERN_TO_CLASS = {
    "center": "Shorts",
    "donut": "Cracks",
    "edgeloc": "LER",
    "edgelocal": "LER",
    "edgering": "LER",
    "local": "Opens",
    "nearfull": "Bridges",
    "none": "Clean",
    "normal": "Clean",
    "random": "Other",
    "scratch": "Scratches",
}


def _images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _wm_class(folder: str) -> str | None:
    key = "".join(ch for ch in folder.lower() if ch.isalpha())
    return WM_PATTERN_TO_CLASS.get(key)


def collect_raw(iesa: Path | None, wm811k: Path | None, multiclass: Path | None) -> list[tuple[Path, str, str]]:
    """Return (path, class, source) for every raw image, using the documented class mapping."""
    items = []
    if iesa:
        for p in _images(iesa):
            top, sub = p.relative_to(iesa).parts[0].lower(), p.parent.name.lower()
            if top == "cracks":
                cls = "Cracks"
            elif top == "vias":
                cls = "Clean" if sub == "clean" else "Vias"
            else:
                cls = "Other"
            items.append((p, cls, "IESA"))
    for root, source in ((wm811k, "WM811K_IMG"), (multiclass, "MULTI_CLASS")):
        if root:
            for p in _images(root):
                wm_cls = _wm_class(p.parent.name)
                if wm_cls:
                    items.append((p, wm_cls, source))
    return items


def out_name(src: Path, source: str) -> str:
    return f"{source}__{src.stem}.png"


def save_processed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("L").resize(IMG_SIZE, Image.Resampling.BILINEAR).save(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iesa", type=Path)
    ap.add_argument("--wm811k", type=Path)
    ap.add_argument("--multiclass", type=Path)
    ap.add_argument("--from-processed", type=Path, help="existing pool of processed images to re-split")
    ap.add_argument("--manifest", type=Path, help="reproduce this exact split")
    ap.add_argument("--write-manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Gather candidates: name -> (source path or None if already processed, class)
    candidates: dict[str, tuple[Path, str, bool]] = {}
    if args.from_processed:
        for p in _images(args.from_processed):
            candidates[p.name] = (p, p.parent.name, False)
    else:
        raw = collect_raw(args.iesa, args.wm811k, args.multiclass)
        if not raw:
            ap.error("no input images: pass --from-processed or at least one of --iesa/--wm811k/--multiclass")
        for p, cls, source in raw:
            candidates[out_name(p, source)] = (p, cls, True)

    if args.manifest:
        items = read_manifest(args.manifest)
        missing = [it.filename for it in items if it.filename not in candidates]
        if missing:
            print(f"warning: {len(missing)} manifest images not found in the sources (e.g. {missing[0]})")
        items = [it for it in items if it.filename in candidates]
    else:
        rng = random.Random(args.seed)
        by_class: dict[str, list[str]] = {c: [] for c in CLASS_NAMES}
        for name, (_, cls, _) in sorted(candidates.items()):
            if cls in by_class:
                by_class[cls].append(name)
        items = []
        for cls, names in by_class.items():
            if not args.from_processed and len(names) > MAX_PER_CLASS:
                # cap large classes, but sample whole groups so augmented siblings stay together
                groups: dict[str, list[str]] = {}
                for n in names:
                    groups.setdefault(group_id(n)[1], []).append(n)
                keys = sorted(groups)
                rng.shuffle(keys)
                names = []
                for k in keys:
                    if len(names) >= MAX_PER_CLASS:
                        break
                    names.extend(groups[k])
            for n in names:
                source, group = group_id(n)
                items.append(Item(n, cls, source, group))
        items = assign_splits(items, seed=args.seed)

    if args.out.exists():
        shutil.rmtree(args.out)
    for it in items:
        src, _, needs_processing = candidates[it.filename]
        dst = args.out / it.split / it.label / it.filename
        if needs_processing:
            save_processed(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if args.write_manifest:
        write_manifest(items, args.write_manifest)

    counts = Counter((it.split, it.label) for it in items)
    print(f"{'class':10s} " + " ".join(f"{s:>6s}" for s in SPLITS))
    for cls in CLASS_NAMES:
        print(f"{cls:10s} " + " ".join(f"{counts[(s, cls)]:6d}" for s in SPLITS))
    print("total      " + " ".join(f"{sum(v for (s, _), v in counts.items() if s == sp):6d}" for sp in SPLITS))


if __name__ == "__main__":
    main()
