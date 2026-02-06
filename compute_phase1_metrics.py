import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

from scripts.infer_phase1 import build_model
from scripts.config_phase1 import CLASS_NAMES
from scripts.dataloaders_phase1 import get_transforms
from torchvision import datasets
from torch.utils.data import DataLoader

DEVICE = torch.device("cpu")
BATCH_SIZE = 64

def load_test_loader():
    test_dir = Path("dataset/test")
    # Use train=False for validation/test transforms
    transforms = get_transforms(train=False)
    test_ds = datasets.ImageFolder(test_dir, transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader

def main():
    model = build_model(num_classes=len(CLASS_NAMES))
    ckpt = torch.load("checkpoints/phase1_best_resnet18.pt", map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()

    test_loader = load_test_loader()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    print("Accuracy:", acc)
    print("Macro Precision:", prec_macro)
    print("Macro Recall:", recall_macro)
    print("Confusion Matrix:")
    print(cm)
    print("Classification report:")
    print(report)

    with open("phase1_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Macro Precision: {prec_macro}\n")
        f.write(f"Macro Recall: {recall_macro}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification report:\n")
        f.write(report)

if __name__ == "__main__":
    main()
