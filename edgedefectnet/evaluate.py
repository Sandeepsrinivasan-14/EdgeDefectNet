"""Evaluate a checkpoint (PyTorch) or exported model (ONNX) on a dataset split.

  python -m edgedefectnet.evaluate --data dataset --split test --model models/edgedefectnet.onnx --out reports

Writes metrics.json (accuracy, macro/weighted F1, per-class report, confusion matrix, per-source accuracy)
and confusion_matrix.png. Class names always come from edgedefectnet.classes.CLASS_NAMES.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .classes import CLASS_NAMES
from .predict import Predictor
from .splits import group_id


def list_split(root: Path, split: str) -> list[tuple[Path, int]]:
    items = []
    for idx, cls in enumerate(CLASS_NAMES):
        folder = root / split / cls
        if folder.is_dir():
            items += [(p, idx) for p in sorted(folder.glob("*.png"))]
    return items


def evaluate(model_path: Path, root: Path, split: str) -> dict:
    predictor = Predictor(model_path)
    items = list_split(root, split)
    y_true = np.array([y for _, y in items])
    y_pred = np.array([predictor.predict_index(p) for p, _ in items])
    sources = np.array([group_id(p.name)[0] for p, _ in items])

    report = classification_report(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    return {
        "model": model_path.name,
        "split": split,
        "n": int(len(items)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": {c: {k: round(float(v), 4) for k, v in report[c].items()} for c in CLASS_NAMES},
        "per_source_accuracy": {
            s: {"n": int(m.sum()), "accuracy": round(float((y_true[m] == y_pred[m]).mean()), 4)}
            for s in sorted(set(sources))
            for m in [sources == s]
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES))).tolist(),
        "labels": CLASS_NAMES,
    }


def plot_confusion(metrics: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion_matrix"])
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ticks = range(len(CLASS_NAMES))
    ax.set_xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(ticks, CLASS_NAMES)
    for i in ticks:
        for j in ticks:
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{metrics['split']} split: accuracy {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model", type=Path, default=Path("models/edgedefectnet.onnx"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()

    metrics = evaluate(args.model, args.data, args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(metrics, args.out / f"confusion_matrix_{args.split}.png")
    print(f"{args.split}: n={metrics['n']} accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}")
    for cls, row in metrics["per_class"].items():
        p, r, f, n = row["precision"], row["recall"], row["f1-score"], int(row["support"])
        print(f"  {cls:10s} precision={p:.3f} recall={r:.3f} f1={f:.3f} n={n}")


if __name__ == "__main__":
    main()
