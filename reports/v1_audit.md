# Audit of the v1 model and results

Version 1 of this repository reported 98.1% test accuracy. Re-running the published ONNX model
(`edge_defect_phase1.onnx`, removed in v2) on the published test split reproduces that number exactly,
including the confusion matrix. Two problems sat behind it.

## 1. Every class name was shifted

`torchvision.datasets.ImageFolder` numbers classes in alphabetical folder order
(`Bridges, Clean, Cracks, LER, Opens, Other, Scratches, Shorts, Vias`), so that is the order of the model's
outputs. The inference script and the metrics script instead named the outputs with a different, hand-written
list (`Clean, Shorts, Opens, Bridges, Vias, Scratches, Cracks, LER, Other`).

Consequences:

- `scripts/infer_phase1.py` printed the wrong class name for almost every image (e.g. a *Bridges* prediction
  was reported as *Clean*).
- The per-class table in the v1 README was misattributed. The row reported as **"Other (precision 0.67,
  support 4)"** was in fact **Vias**, the only class with 4 test images; the confusion-matrix row supports
  `[132, 134, 156, 225, 133, 133, 133, 133, 4]` match the alphabetical folder counts exactly.

v2 keeps a single alphabetical `CLASS_NAMES` list used by every component, and a test asserts it matches
ImageFolder's order.

## 2. Train/test leakage

The largest source is a Roboflow export in which each original wafer map appears several times as augmented
copies (`image_Center_12116_png_jpg.rf.<hash>.png`). v1 split at the file level, so copies of the same wafer
landed in both training and test data.

| v1 test subset | Images | Accuracy | Macro-F1 |
|---|---|---|---|
| All test images | 1,183 | 98.14% | 0.961 |
| Images whose original wafer also appears in train or validation | 476 (40%) | 100.00% | 1.000 |
| Images whose original wafer is genuinely unseen | 707 | 96.89% | 0.949 |

| v1 test accuracy by source | Images | Accuracy |
|---|---|---|
| MULTI_CLASS (Roboflow, 640x640 renderings) | 1,028 | 99.81% |
| IESA | 27 | 92.59% |
| WM811K_IMG (32x32 wafer maps) | 128 | 85.94% |

The honest held-out accuracy of the v1 model was therefore about 97%, not 98.1%, and markedly lower on the
WM-811K images.

v2 assigns every copy of an original wafer to the same split (`edgedefectnet/splits.py`), publishes the exact
split in `data/manifest.csv` (a test verifies no group crosses splits) and retrains from ImageNet weights.
See the README for the v2 results.
