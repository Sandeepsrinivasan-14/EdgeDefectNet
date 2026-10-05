"""Inference with the exported ONNX model (needs only onnxruntime, numpy and Pillow).

python -m edgedefectnet.predict examples/*.png
python -m edgedefectnet.predict wafer.png --model models/edgedefectnet.onnx --top 3 --json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classes import CLASS_NAMES
from .preprocessing import load_grayscale, to_input_array

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "edgedefectnet.onnx"


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Predictor:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"model not found: {model_path}")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        n_out = self.session.get_outputs()[0].shape[-1]
        if n_out != len(CLASS_NAMES):
            raise ValueError(f"model has {n_out} outputs but there are {len(CLASS_NAMES)} classes")

    def logits(self, image) -> np.ndarray:
        img = load_grayscale(image) if isinstance(image, (str, Path)) else image
        return self.session.run(None, {self.input_name: to_input_array(img)})[0][0]

    def predict_index(self, image) -> int:
        return int(self.logits(image).argmax())

    def predict(self, image) -> Prediction:
        probs = softmax(self.logits(image))
        idx = int(probs.argmax())
        return Prediction(CLASS_NAMES[idx], float(probs[idx]), {c: float(p) for c, p in zip(CLASS_NAMES, probs)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="print one JSON object per image")
    args = ap.parse_args()

    predictor = Predictor(args.model)
    for path in args.images:
        pred = predictor.predict(path)
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


if __name__ == "__main__":
    main()
