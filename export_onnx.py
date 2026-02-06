import os
import sys
import torch

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Reuse build_model from scripts.infer_phase1 (relative import in that file)
from scripts.infer_phase1 import build_model

NUM_CLASSES = 9
H, W = 224, 224  # set to your real input size if different

model = build_model(num_classes=NUM_CLASSES)

# Use your real checkpoint file
state_path = os.path.join(ROOT, "checkpoints", "phase1_best_resnet18.pt")
state = torch.load(state_path, map_location="cpu")
model.load_state_dict(state)

model.eval()

dummy = torch.randn(1, 1, H, W)  # 1 channel, because build_model replaces conv1 for grayscale
onnx_path = os.path.join(ROOT, "edge_defect_phase1.onnx")
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["input"],
    output_names=["logits"],
    opset_version=18,
)
print(f"Exported ONNX to {onnx_path}")
