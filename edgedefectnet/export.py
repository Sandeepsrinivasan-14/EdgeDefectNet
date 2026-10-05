"""Export a trained checkpoint to a single self-contained ONNX file and verify it matches PyTorch.

python -m edgedefectnet.export --checkpoint checkpoints/best.pt --out models/edgedefectnet.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import load_checkpoint
from .preprocessing import IMAGE_SIZE


def export(checkpoint: Path, out: Path, opset: int = 17) -> float:
    model = load_checkpoint(checkpoint)
    dummy = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 1, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    if not np.allclose(ref, got, atol=1e-3):
        raise RuntimeError(f"ONNX output differs from PyTorch (max abs diff {max_diff})")
    return max_diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/edgedefectnet.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    diff = export(args.checkpoint, args.out, args.opset)
    size_mb = args.out.stat().st_size / 1e6
    print(f"exported {args.out} ({size_mb:.1f} MB); max |torch - onnx| = {diff:.2e}")


if __name__ == "__main__":
    main()
