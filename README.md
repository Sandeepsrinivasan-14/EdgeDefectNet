# EdgeDefectNet

[![CI](https://github.com/Sandeepsrinivasan-14/EdgeDefectNet/actions/workflows/ci.yml/badge.svg)](https://github.com/Sandeepsrinivasan-14/EdgeDefectNet/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Model](https://img.shields.io/badge/model-ResNet--18%20%7C%20ONNX-orange)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Wafer defect classification with a grayscale ResNet-18. Nine classes, a leakage-free train/validation/test split,
an ONNX model you can run with three small dependencies, and a web app that runs the model **in your browser**.

![Web app: the selected wafer map rendered on a 3D wafer](docs/web_hero.png)

## Results (held-out test split)

| Metric | Value |
|---|---|
| Accuracy | **97.96%** (1,176 images) |
| Macro-F1 | 0.981 |
| Weighted-F1 | 0.980 |
| Validation accuracy / macro-F1 (model selection) | 97.45% / 0.966 |

| Class | Precision | Recall | F1 | Test images |
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
| Roboflow wafer-map renderings (`MULTI_CLASS`) | 1,027 | 99.6% |
| IESA challenge images | 29 | 100% |
| WM-811K 32x32 wafer maps | 120 | **83.3%** |

How to read these numbers:

- The test split was used once, after training finished; model selection used validation macro-F1 only.
- The low-resolution WM-811K images are clearly harder than the Roboflow renderings. Most errors come from them.
- *Vias* has 4 test images, so its perfect score carries almost no information. The macro-F1 is pulled up by it;
  accuracy and weighted-F1 are the more reliable summaries.
- Full metrics: [`reports/metrics_test.json`](reports/metrics_test.json),
  [confusion matrix](reports/confusion_matrix_test.png), [training curves](reports/training_curves.png).

### What changed from v1

v1 reported 98.1% accuracy, but an [audit](reports/v1_audit.md) found two problems:

1. **Shifted class names.** The model outputs classes in alphabetical (ImageFolder) order, but inference and the
   metrics report named them with a different list, so predictions and the per-class table were mislabelled.
2. **Train/test leakage.** Augmented copies of the same wafer appeared in both training and test data. On test
   images whose wafer was genuinely unseen, v1 scored 96.9%.

v2 uses one class list everywhere (guarded by a test), splits by original wafer so no copy crosses splits
(`data/manifest.csv`, also tested), and retrains from ImageNet weights. Its 97.96% is measured on a split with
no leakage at all.

## Web app

`web/` is a static site: pick a held-out example or drop an image, and the ONNX model runs locally through
ONNX Runtime Web (WebAssembly). The selected wafer map is rendered onto a 3D wafer (three.js) with a thin-film
iridescence shader and an amber "exposure" scan when a prediction completes. The model card section reads the
real test metrics and draws the confusion matrix.

| Prediction | Model card |
|---|---|
| ![Prediction panel](docs/web_prediction.png) | ![Model card](docs/web_model_card.png) |

```bash
python scripts/build_site.py              # assembles _site/ (web app + model + examples + metrics)
python -m http.server -d _site 8000       # open http://localhost:8000
```

The **Web app (GitHub Pages)** workflow deploys `_site/` on every push to `main`. One-time setup: repository
**Settings → Pages → Source: GitHub Actions**. A browser test (`tests/test_web.py`) checks that the web app's
predictions match the Python predictor on every example.

## Quick start (inference)

```bash
pip install -r requirements.txt            # onnxruntime, numpy, pillow
python -m edgedefectnet.predict examples/*.png
python -m edgedefectnet.predict my_wafer.png --top 3 --json
```

```text
Bridges__MULTI_CLASS.png: Bridges (99.8%)  [next: Other 0.2%, LER 0.0%]
Cracks__IESA.png: Cracks (100.0%)  [next: Bridges 0.0%, Opens 0.0%]
Other__MULTI_CLASS.png: Other (99.0%)  [next: Bridges 1.0%, Cracks 0.0%]
```

From Python:

```python
from edgedefectnet.predict import Predictor

pred = Predictor().predict("my_wafer.png")
print(pred.label, pred.confidence, pred.probabilities)
```

There is also a small Gradio demo: `pip install gradio && python app.py`.

## Classes and data

The nine classes come from the IESA challenge: Bridges, Clean, Cracks, LER, Opens, Other, Scratches, Shorts, Vias.
Most images are WM-811K-style wafer maps whose spatial failure patterns were mapped onto these names (for
example *Center* to *Shorts*, *Donut* to *Cracks*, *Edge-Ring* and *Edge-Loc* to *LER*). For those images the
class names are **proxy labels**: the model recognises the spatial pattern, not the electrical defect. The full
mapping, sources and rebuild instructions are in [`data/README.md`](data/README.md).

Images are not stored in Git. `data/manifest.csv` lists every image with its split, class, source and wafer group,
and `scripts/build_dataset.py` rebuilds the dataset from the raw sources.

## Training

```bash
pip install -r requirements-train.txt
# ImageNet-pretrained ResNet-18 in torchvision format (timm's ResNet-18 A1 weights)
curl -L -o weights/resnet18.pth --create-dirs \
  https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth

python -m edgedefectnet.train --data dataset --imagenet-weights weights/resnet18.pth --out checkpoints --epochs 10
python -m edgedefectnet.export --checkpoint checkpoints/best.pt --out models/edgedefectnet.onnx
python -m edgedefectnet.evaluate --data dataset --split test --model models/edgedefectnet.onnx --out reports
```

Setup used for the published model: 1-channel ResNet-18 (ImageNet filters averaged to grayscale), 224x224 input,
AdamW (lr 3e-4, weight decay 1e-4) with cosine decay, batch 32, class-weighted cross-entropy (weights capped at
10), light augmentation (rotation ±10°, crop 90-100%, horizontal flip), seed 42, 10 epochs on CPU (about 8
minutes per epoch). The best validation checkpoint was epoch 10. The run was interrupted once by a machine
restart after epoch 2 and resumed from that checkpoint with a fresh optimizer state; `--resume` now saves full
optimizer and scheduler state every epoch.

## Project layout

```text
edgedefectnet/        package: classes, preprocessing, splits, model, data, train, evaluate, export, predict
models/               edgedefectnet.onnx (single file, dynamic batch, opset 17)
web/                  browser app (HTML/CSS/JS, self-hosted Archivo font)
examples/             one held-out test image per class
data/                 manifest.csv and dataset documentation
reports/              test/validation metrics, confusion matrices, training history, v1 audit
scripts/              build_dataset.py, build_site.py, plot_history.py
tests/                classes and split leakage, inference, ONNX export parity, browser parity
app.py                Gradio demo
```

## Development

```bash
pip install -r requirements-dev.txt
pytest -q                     # torch and browser tests are skipped if those packages are missing
ruff check . && ruff format --check .
```

## License

[MIT](LICENSE) © Sandeep Srinivasan. The Archivo font is under the SIL Open Font License (`web/fonts/OFL-Archivo.txt`).
Check each dataset source's licence before redistributing images.
