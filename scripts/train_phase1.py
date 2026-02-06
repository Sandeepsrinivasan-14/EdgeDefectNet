import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from .config_phase1 import CLASS_NAMES, DATASET_ROOT, IF_TO_FINAL
from .dataloaders_phase1 import get_dataloaders
from .class_weights_phase1 import get_train_counts


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    conv1 = model.conv1
    new_conv1 = nn.Conv2d(
        1,
        conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        new_conv1.weight[:] = conv1.weight.mean(dim=1, keepdim=True)
    model.conv1 = new_conv1

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(train_ds):
    counts, folder_classes = get_train_counts()
    total = sum(counts.values())

    weights_if_order = []
    for name in folder_classes:
        n = counts.get(name, 0)
        w = total / n if n > 0 else 0.0
        weights_if_order.append(w)

    print("Class weights in ImageFolder order:")
    for name, w in zip(folder_classes, weights_if_order):
        print(f"  {name:10s}: {w:.2f}")

    return torch.tensor(weights_if_order, dtype=torch.float32)


def train_model(epochs: int = 25, lr: float = 1e-3, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = get_dataloaders(
        batch_size=batch_size, num_workers=2
    )

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    best_path = Path(DATASET_ROOT).parent / "checkpoints"
    best_path.mkdir(parents=True, exist_ok=True)
    best_model_file = best_path / "phase1_best_resnet18.pt"

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        t0 = time.time()

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = correct / total
        scheduler.step(val_loss)

        dt = time.time() - t0
        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}, time: {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_file)
            print("  -> New best model saved.")

    print("\nTraining done. Best val acc:", best_val_acc)

    # ---- TEST EVALUATION ----
    print("\nEvaluating best model on TEST set...")
    model.load_state_dict(torch.load(best_model_file, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    num_if_classes = len(test_ds.classes)
    per_if_correct = [0] * num_if_classes
    per_if_total = [0] * num_if_classes

    with torch.no_grad():
        for images, labels in test_dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for l, p in zip(labels, preds):
                per_if_total[l.item()] += 1
                if l == p:
                    per_if_correct[l.item()] += 1

    test_acc = correct / total
    print(f"TEST accuracy: {test_acc:.4f}")

    print("Per-class accuracy (mapped to CLASS_NAMES):")
    folder_classes = test_ds.classes
    for if_idx, name_if in enumerate(folder_classes):
        tot = per_if_total[if_idx]
        if tot == 0:
            acc = 0.0
        else:
            acc = per_if_correct[if_idx] / tot

        final_idx = IF_TO_FINAL[if_idx]
        final_name = CLASS_NAMES[final_idx]
        print(f"  IF:{name_if:10s} -> {final_name:10s}: {acc:.4f} ({per_if_correct[if_idx]}/{tot})")


if __name__ == "__main__":
    train_model(epochs=25, lr=1e-3, batch_size=32)
