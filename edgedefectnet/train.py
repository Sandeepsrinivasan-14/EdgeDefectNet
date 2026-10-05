"""Train EdgeDefectNet.

  python -m edgedefectnet.train --data dataset --imagenet-weights weights/resnet18.pth --out checkpoints

Model selection uses the validation split only (macro-F1). The test split is never touched here;
run `python -m edgedefectnet.evaluate` on the best checkpoint afterwards.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .data import class_weights, image_folder, make_loader
from .model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item()
        n += y.numel()
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), loss_sum / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--imagenet-weights", type=Path, help="torchvision-format ResNet-18 ImageNet weights")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3, help="early-stop after N epochs without val improvement")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if it exists")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = image_folder(args.data, "train", train=True)
    val_ds = image_folder(args.data, "val")
    train_dl = make_loader(train_ds, args.batch_size, True, args.workers)
    val_dl = make_loader(val_ds, args.batch_size * 2, False, args.workers)

    model = build_model(imagenet_weights=args.imagenet_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best_f1, stale, start = [], -1.0, 0, 1
    last_path = args.out / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        history, best_f1, stale, start = state["history"], state["best_f1"], state["stale"], state["epoch"] + 1
        print(f"resumed from {last_path} after epoch {state['epoch']}", flush=True)

    print(f"device={device} train={len(train_ds)} val={len(val_ds)}", flush=True)
    for epoch in range(start, args.epochs + 1):
        model.train()
        t0, loss_sum, seen = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.numel()
            seen += y.numel()
        scheduler.step()

        y_true, y_pred, val_loss = predict(model, val_dl, device)
        val_acc = float((y_true == y_pred).mean())
        val_f1 = float(f1_score(y_true, y_pred, average="macro"))
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "args": vars(args)
                    | {"data": str(args.data), "out": str(args.out), "imagenet_weights": str(args.imagenet_weights)},
                },
                args.out / "best.pt",
            )
        else:
            stale += 1
        # Saved every epoch so an interrupted run can continue with --resume.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "epoch": epoch,
                "history": history,
                "best_f1": best_f1,
                "stale": stale,
            },
            last_path,
        )
        if stale >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val macro-F1 {best_f1:.4f}; checkpoint at {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
