"""Inference with the exported ONNX model (needs only onnxruntime, numpy and Pillow).

python -m edgedefectnet.predict examples/*.png
python -m edgedefectnet.predict wafer.png --model models/edgedefectnet.onnx --top 3 --json

The model file is looked up in this order: the ``model_path`` argument / ``--model`` flag, the
``EDGEDEFECTNET_MODEL`` environment variable, then ``models/edgedefectnet.onnx`` in a source checkout.
Exit status of the CLI: 0 all images classified, 1 at least one image was unreadable, 2 the model could not be loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from . import __version__
from .classes import CLASS_NAMES
from .preprocessing import InvalidImageError, load_grayscale, to_input_array

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Environment variable that overrides the default model location.
MODEL_ENV_VAR = "EDGEDEFECTNET_MODEL"
#: Path of the model in a source checkout.
CHECKOUT_MODEL = Path(__file__).resolve().parents[1] / "models" / "edgedefectnet.onnx"
DEFAULT_MODEL = CHECKOUT_MODEL

ImageInput = str | os.PathLike[str] | Image.Image


class ModelError(RuntimeError):
    """The model file is corrupt, has the wrong shape, or fails its integrity check."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def softmax(logits: NDArray[np.floating]) -> NDArray[np.floating]:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def expected_sha256(model_path: str | Path) -> str | None:
    """Read the digest from the ``<model>.sha256`` sidecar file (``sha256sum`` format), if it exists."""
    sidecar = Path(str(model_path) + ".sha256")
    if not sidecar.is_file():
        return None
    parts = sidecar.read_text(encoding="utf-8").split()
    return parts[0].lower() if parts else None


def verify_model(model_path: str | Path, expected: str | None = None) -> str:
    """Check the model's SHA-256 against ``expected`` or the sidecar file; returns the digest."""
    expected = expected or expected_sha256(model_path)
    if not expected:
        raise ModelError(f"no expected SHA-256 for {model_path} (pass one or provide {model_path}.sha256)")
    actual = sha256_file(model_path)
    if actual != expected.lower():
        raise ModelError(f"SHA-256 mismatch for {model_path}: expected {expected}, got {actual}")
    return actual


def _check_model_file(model_path: Path) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"model not found: {model_path}. Run from a source checkout, download edgedefectnet.onnx from the GitHub "
            f"release, or set {MODEL_ENV_VAR} to its location."
        )
    with open(model_path, "rb") as fh:
        head = fh.read(64)
    if not head:
        raise ModelError(f"model file is empty: {model_path}")
    if head.startswith(b"version https://git-lfs"):
        raise ModelError(f"{model_path} is a Git LFS pointer, not the model. Run `git lfs pull` or download the release asset.")


class Predictor:
    """Wafer-image classifier backed by an ONNX Runtime CPU session.

    Args:
        model_path: ONNX file; defaults to ``$EDGEDEFECTNET_MODEL`` or ``models/edgedefectnet.onnx``.
        verify: check the file's SHA-256 against the ``<model>.sha256`` sidecar before loading.
    """

    def __init__(self, model_path: str | Path | None = None, *, verify: bool = False) -> None:
        import onnxruntime as ort

        path = Path(model_path) if model_path is not None else Path(os.environ.get(MODEL_ENV_VAR) or CHECKOUT_MODEL)
        _check_model_file(path)
        if verify:
            verify_model(path)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        except Exception as exc:  # onnxruntime raises several unrelated exception types for bad files
            raise ModelError(f"cannot load model {path}: {exc}") from exc
        self.model_path = path
        self.input_name: str = self.session.get_inputs()[0].name
        n_out = self.session.get_outputs()[0].shape[-1]
        if n_out != len(CLASS_NAMES):
            raise ModelError(f"model has {n_out} outputs but there are {len(CLASS_NAMES)} classes")

    def logits(self, image: ImageInput) -> NDArray[np.float32]:
        if isinstance(image, (str, os.PathLike)):
            img = load_grayscale(Path(image))
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError(f"expected a file path or PIL.Image, got {type(image).__name__}")
        out = self.session.run(None, {self.input_name: to_input_array(img)})[0][0]
        if not np.isfinite(out).all():
            raise ModelError("model produced non-finite outputs")
        return np.asarray(out, dtype=np.float32)

    def predict_index(self, image: ImageInput) -> int:
        return int(np.argmax(self.logits(image)))

    def predict(self, image: ImageInput) -> Prediction:
        probs = softmax(self.logits(image))
        idx = int(np.argmax(probs))
        return Prediction(CLASS_NAMES[idx], float(probs[idx]), {c: float(p) for c, p in zip(CLASS_NAMES, probs)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument(
        "--model", type=Path, default=None, help=f"ONNX model (default: ${MODEL_ENV_VAR} or models/edgedefectnet.onnx)"
    )
    ap.add_argument("--top", type=int, default=3, help=f"number of classes to report (1-{len(CLASS_NAMES)})")
    ap.add_argument("--json", action="store_true", help="print one JSON object per image")
    ap.add_argument("--verify", action="store_true", help="check the model's SHA-256 against its .sha256 sidecar first")
    ap.add_argument("--version", action="version", version=f"edgedefectnet {__version__}")
    args = ap.parse_args(argv)
    if not 1 <= args.top <= len(CLASS_NAMES):
        ap.error(f"--top must be between 1 and {len(CLASS_NAMES)}")

    try:
        predictor = Predictor(args.model, verify=args.verify)
    except (FileNotFoundError, ModelError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    failures = 0
    for path in args.images:
        try:
            pred = predictor.predict(path)
        except (FileNotFoundError, InvalidImageError) as exc:
            failures += 1
            if args.json:
                print(json.dumps({"image": str(path), "error": str(exc)}))
            else:
                print(f"error: {exc}", file=sys.stderr)
            continue
        top = sorted(pred.probabilities.items(), key=lambda kv: -kv[1])[: args.top]
        if args.json:
            print(
                json.dumps(
                    {
                        "image": str(path),
                        "label": pred.label,
                        "confidence": round(pred.confidence, 4),
                        "top": {k: round(v, 4) for k, v in top},
                    }
                )
            )
        else:
            alt = ", ".join(f"{k} {v:.1%}" for k, v in top[1:])
            print(f"{path.name}: {pred.label} ({pred.confidence:.1%})" + (f"  [next: {alt}]" if alt else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
