import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T, models

from .config_phase1 import CLASS_NAMES, NORM_MEAN, NORM_STD
from .dataloaders_phase1 import get_transforms


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


def load_model(checkpoint_path: Path, device: torch.device):
    model = build_model(num_classes=len(CLASS_NAMES))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path):
    img = Image.open(image_path).convert("RGB")
    # use same val/test transforms but without random ops
    transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[NORM_MEAN], std=[NORM_STD]),
    ])
    tensor = transform(img).unsqueeze(0)  # [1, 1, 224, 224]
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path("checkpoints") / "phase1_resnet18_v1_98p1.pt"),
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)

    if not image_path.is_file():
        print(f"ERROR: image file not found: {image_path}")
        return
    if not ckpt_path.is_file():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = load_model(ckpt_path, device)
    x = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)

    top_prob, top_idx = torch.max(probs, dim=0)
    pred_class = CLASS_NAMES[top_idx.item()]

    print(f"Predicted class: {pred_class} (index {top_idx.item()})")
    print("Probabilities:")
    for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
        print(f"  {i:2d} {name:10s}: {p.item():.4f}")


if __name__ == "__main__":
    main()
