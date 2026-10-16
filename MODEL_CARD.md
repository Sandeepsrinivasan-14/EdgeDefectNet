# Model card: EdgeDefectNet v2.0.0

## Summary

| | |
|---|---|
| Task | Single-label image classification of wafer images into 9 defect classes |
| Architecture | ResNet-18, first convolution replaced by a 1-channel version (ImageNet filters averaged to grayscale), 9-way linear head |
| Input | Any image Pillow can read; converted to 8-bit grayscale, resized to 224x224 (bilinear), normalised with mean 0.45 / std 0.23 |
| Output | 9 logits in alphabetical class order: Bridges, Clean, Cracks, LER, Opens, Other, Scratches, Shorts, Vias |
| Format | ONNX (opset 17, dynamic batch), `models/edgedefectnet.onnx`, 44.7 MB |
| SHA-256 | `2d5c927696d89316bba623a4feda4917c444ee70eafeefe5cfee1f3cc9c2e5d4` (also in `models/edgedefectnet.onnx.sha256`, checked by a test) |
| Runtime | ONNX Runtime on CPU (Python) or ONNX Runtime Web (browser, WebAssembly) |
| License | MIT (code and weights). Dataset licences are separate; see below. |

## Intended use

- Demonstration, research and education: classifying wafer-map style images into the nine challenge classes and
  studying leakage-free evaluation on this dataset.
- A starting point (weights, preprocessing, export and test scaffolding) for training on your own inspection images.

## Out-of-scope use

- Production yield control, scrap/ship decisions or any safety-relevant process decision. The model has not been
  validated on images from a real fab line, tool or inspection system.
- Inputs that are not wafer images. The model always returns one of the nine classes; the softmax confidence is not
  calibrated and there is no out-of-distribution detection, so an unrelated image can still get a confident label.

## Training data

About 7,800 images assembled by `scripts/build_dataset.py` from three public sources (list, mapping and rebuild
instructions in [`data/README.md`](data/README.md); the per-image split is in `data/manifest.csv`):

| Source | Role |
|---|---|
| Roboflow wafer-map renderings (`MULTI_CLASS`) | Largest source; each original wafer appears as several augmented copies |
| WM-811K 32x32 wafer maps (`WM811K_IMG`) | Low-resolution wafer-level failure patterns |
| IESA challenge images (`IESA`) | Few images; the only source of real *Vias* examples |

Images are not stored in the repository. Splitting is by original wafer (all augmented copies of one wafer stay in
the same split) at a 70/15/15 ratio, stratified by class; `tests/test_classes_and_splits.py` asserts that no group
crosses splits.

**Proxy labels.** Most images are WM-811K-style wafer maps whose *spatial failure pattern* was mapped onto the nine
challenge names (for example *Center* to *Shorts*, *Donut* to *Cracks*, *Edge-Ring* and *Edge-Loc* to *LER*). For
those images the class name is a proxy: the model recognises the spatial pattern, not the electrical defect named
by the label.

## Training procedure

AdamW (lr 3e-4, weight decay 1e-4), cosine schedule, batch 32, class-weighted cross-entropy (weights capped at 10),
rotation +/-10 degrees, random resized crop (90-100%), horizontal flip, seed 42, 10 epochs on CPU. Initialised from
timm ResNet-18 A1 ImageNet weights. Model selection used validation macro-F1 only; the test split was evaluated once
after training. Details and the exact commands are in the README.

## Evaluation (held-out test split, 1,176 images)

| Metric | Value |
|---|---|
| Accuracy | 97.96% |
| Macro-F1 | 0.981 |
| Weighted-F1 | 0.980 |
| Validation accuracy / macro-F1 | 97.45% / 0.966 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bridges | 0.970 | 1.000 | 0.985 | 131 |
| Clean | 0.971 | 1.000 | 0.985 | 133 |
| Cracks | 0.975 | 0.994 | 0.984 | 155 |
| LER | 0.996 | 0.991 | 0.993 | 225 |
| Opens | 0.977 | 0.955 | 0.966 | 132 |
| Other | 0.984 | 0.955 | 0.969 | 132 |
| Scratches | 0.977 | 0.947 | 0.962 | 132 |
| Shorts | 0.977 | 0.985 | 0.981 | 132 |
| Vias | 1.000 | 1.000 | 1.000 | 4 |

| Accuracy by source | Images | Accuracy |
|---|---|---|
| `MULTI_CLASS` (Roboflow renderings) | 1,027 | 99.6% |
| `IESA` | 29 | 100% |
| `WM811K_IMG` (32x32) | 120 | **83.3%** |

Machine-readable results: [`reports/metrics_test.json`](reports/metrics_test.json),
[`reports/metrics_val.json`](reports/metrics_val.json); plots in `reports/`.

## Limitations and risks

- **Source shift is the dominant error.** Accuracy on low-resolution WM-811K images (83.3%) is far below the
  Roboflow renderings (99.6%). Expect a similar drop on any new image source.
- **Vias is almost untested** (4 test images). Its 1.000 scores carry almost no information and inflate macro-F1;
  prefer accuracy and weighted-F1.
- **Near-duplicate structure.** The leading source contains augmented copies of the same wafers. The split is by
  wafer, but wafers from the same upstream collection still share rendering style, so scores measure
  in-distribution generalisation, not transfer to a new fab or tool.
- **Proxy labels** (above): good performance does not imply the model can identify the named electrical defects.
- **Version 1 of this project was wrong** about its own labels and leaked test data; see
  [`reports/v1_audit.md`](reports/v1_audit.md). The v1 model is not distributed.

## Maintenance

Retraining changes the ONNX file. When that happens update, in one commit: `models/edgedefectnet.onnx.sha256`, the
reports, the numbers in this card and the README, and `CHANGELOG.md`. A test fails if the checksum file and the model
disagree.
