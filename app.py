"""Gradio demo: upload a wafer image and get the predicted class with probabilities.

pip install -e ".[demo]"
python app.py            # http://127.0.0.1:7860

Environment variables:
  EDGEDEFECTNET_MODEL   path to the ONNX model (default: models/edgedefectnet.onnx)
  HOST / PORT           bind address (default 127.0.0.1:7860; the Docker image uses 0.0.0.0)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from edgedefectnet.predict import ModelError, Predictor
from edgedefectnet.preprocessing import InvalidImageError

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("edgedefectnet.app")


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Load the model once, on first use, so importing this module is cheap and failures surface as UI errors."""
    return Predictor()


def classify(image: Image.Image | None) -> dict[str, float]:
    """Return {class: probability} for a wafer image; an empty dict clears the output."""
    import gradio as gr

    if image is None:
        return {}
    try:
        return get_predictor().predict(image).probabilities
    except InvalidImageError as exc:
        raise gr.Error(f"Could not use that image: {exc}") from exc
    except (FileNotFoundError, ModelError) as exc:
        log.exception("model unavailable")
        raise gr.Error(f"The model is unavailable: {exc}") from exc
    except Exception as exc:
        log.exception("unexpected inference failure")
        raise gr.Error("Prediction failed unexpectedly; see the server log.") from exc


def build_demo() -> Any:
    import gradio as gr

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", image_mode="L", label="Wafer image"),
        outputs=gr.Label(num_top_classes=3, label="Prediction"),
        examples=[[str(p)] for p in sorted((ROOT / "examples").glob("*.png"))],
        title="EdgeDefectNet",
        description="Grayscale ResNet-18 classifying wafer images into 9 defect classes. "
        "See the README for how the classes map to the source datasets.",
        flagging_mode="never",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    get_predictor()  # fail fast at start-up if the model is missing or corrupt
    build_demo().launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
