"""Plot training curves from history.json.

python scripts/plot_history.py checkpoints/history.json reports/training_curves.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(src: str, dst: str) -> None:
    with open(src, encoding="utf-8") as fh:
        hist = json.load(fh)
    ep = [h["epoch"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    a.plot(ep, [h["train_loss"] for h in hist], marker="o", label="train")
    a.plot(ep, [h["val_loss"] for h in hist], marker="o", label="validation")
    a.set_title("Loss")
    a.set_xlabel("epoch")
    a.legend()
    b.plot(ep, [h["val_acc"] for h in hist], marker="o", label="val accuracy")
    b.plot(ep, [h["val_macro_f1"] for h in hist], marker="o", label="val macro-F1")
    b.set_title("Validation metrics")
    b.set_xlabel("epoch")
    b.legend()
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
