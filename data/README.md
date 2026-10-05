# Dataset

Images are not stored in Git (v1 committed the same ~7,900 images three times, 644 MB). This folder holds
`manifest.csv`, which defines the exact split used for the published model:

| column | meaning |
|---|---|
| `split` | `train`, `val` or `test` |
| `label` | one of the 9 classes in `edgedefectnet/classes.py` |
| `source` | `MULTI_CLASS`, `WM811K_IMG` or `IESA` |
| `group` | original wafer id; all augmented copies of one wafer share it and always sit in the same split |
| `filename` | processed file name, `<SOURCE>__<original stem>.png` |

## Sources

| Source | What it is | Images used |
|---|---|---|
| `MULTI_CLASS` | Roboflow export of WM-811K-style wafer-map renderings (640x640) with augmented copies, 9 patterns | 6,777 |
| `WM811K_IMG` | 32x32 image subset of the WM-811K wafer map dataset, 9 patterns | 873 |
| `IESA` | Preview images from the IESA challenge: `Cracks/Defect`, `Vias/Defect`, `Vias/Clean` | 183 |

WM-811K: M.-J. Wu, J.-S. R. Jang and J.-L. Chen, "Wafer Map Failure Pattern Recognition and Similarity Ranking
for Large-Scale Data Sets", *IEEE Transactions on Semiconductor Manufacturing*, 28(1), 2015.
Check each source's licence before redistributing images.

## Class mapping

The challenge defines 9 die-level defect classes. The public sources describe wafer-level spatial failure
patterns, so their patterns were mapped onto the challenge classes as follows. For those sources the class
names are **proxy labels**: the model learns the spatial pattern, not the electrical defect named.

| Source pattern | Class |
|---|---|
| Center | Shorts |
| Donut | Cracks |
| Edge-Loc, Edge-Ring | LER |
| Local | Opens |
| Near-Full | Bridges |
| None / Normal | Clean |
| Random | Other |
| Scratch | Scratches |
| IESA Cracks/Defect | Cracks |
| IESA Vias/Defect | Vias |
| IESA Vias/Clean | Clean |

`Vias` therefore has only 23 images in total (4 in test), so its metrics are very uncertain.

## Rebuilding

```bash
# exact published split from the raw source folders
python scripts/build_dataset.py --manifest data/manifest.csv \
    --iesa raw/IESA_unique --wm811k raw/wm811k_images --multiclass raw/multi_class_wafer --out dataset

# or a fresh group-aware split (writes a new manifest)
python scripts/build_dataset.py --iesa ... --wm811k ... --multiclass ... --out dataset --write-manifest my_manifest.csv
```

Output: `dataset/{train,val,test}/<class>/*.png`, 224x224 8-bit grayscale.
