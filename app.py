"""Gradio demo: upload a wafer image and get the predicted class, a confidence gauge and per-class probabilities.

pip install -e ".[demo]"
python app.py            # http://127.0.0.1:7860

Environment variables:
  EDGEDEFECTNET_MODEL   path to the ONNX model (default: models/edgedefectnet.onnx)
  HOST / PORT           bind address (default 127.0.0.1:7860; the Docker image uses 0.0.0.0)

The page styling is plain CSS in app.css (no external fonts or scripts), so it works offline and inside the Docker image.
"""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

from PIL import Image

from edgedefectnet.classes import CLASS_INFO, CLASS_NAMES
from edgedefectnet.predict import ModelError, Prediction, Predictor
from edgedefectnet.preprocessing import InvalidImageError

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("edgedefectnet.app")

EXAMPLES_DIR = ROOT / "examples"
GAUGE_RADIUS = 54
GAUGE_CIRCUMFERENCE = 2 * math.pi * GAUGE_RADIUS


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Load the model once, on first use, so importing this module is cheap and failures surface as UI errors.

    The demo only ever classifies in-memory images (``predict_image``), so no upload path reaches the file system.
    """
    return Predictor()


def _predict(image: Image.Image) -> Prediction:
    import gradio as gr

    try:
        return get_predictor().predict_image(image)
    except InvalidImageError as exc:
        raise gr.Error(f"Could not use that image: {exc}") from exc
    except (FileNotFoundError, ModelError) as exc:
        log.exception("model unavailable")
        raise gr.Error(f"The model is unavailable: {exc}") from exc
    except Exception as exc:
        log.exception("unexpected inference failure")
        raise gr.Error("Prediction failed unexpectedly; see the server log.") from exc


def classify(image: Image.Image | None) -> dict[str, float]:
    """Return {class: probability} for a wafer image; an empty dict clears the output."""
    if image is None:
        return {}
    return _predict(image).probabilities


# ---------------------------------------------------------------- presentation


def render_result(probabilities: dict[str, float]) -> str:
    """HTML for the result card: confidence gauge, ranked bars and an explainer for the winning class.

    Every dynamic value is either a class name from CLASS_NAMES or a number, but text is escaped anyway.
    """
    if not probabilities:
        return (
            '<div class="edn-empty"><div class="edn-empty-ring" aria-hidden="true"></div>'
            "<p>Drop a wafer image or pick an example to see the prediction.</p></div>"
        )
    ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
    best, conf = ranked[0]
    info = CLASS_INFO[best]
    dash = GAUGE_CIRCUMFERENCE * conf
    bars = "".join(
        f'<li style="--c:{CLASS_INFO[name]["color"]}"><span class="edn-name">{escape(name)}</span>'
        f'<span class="edn-track"><span class="edn-fill" style="width:{max(p * 100, 0.6):.2f}%;'
        f'animation-delay:{i * 45}ms"></span></span><span class="edn-pct">{p * 100:.1f}%</span></li>'
        for i, (name, p) in enumerate(ranked)
    )
    return f"""
<div class="edn-result" role="status" aria-live="polite" style="--c:{info["color"]}">
  <div class="edn-top">
    <svg class="edn-gauge" viewBox="0 0 128 128" role="img" aria-label="Confidence {conf * 100:.1f} percent">
      <circle class="edn-gauge-bg" cx="64" cy="64" r="{GAUGE_RADIUS}"/>
      <circle class="edn-gauge-fg" cx="64" cy="64" r="{GAUGE_RADIUS}"
        stroke-dasharray="{dash:.2f} {GAUGE_CIRCUMFERENCE:.2f}"/>
      <text x="64" y="72" text-anchor="middle" class="edn-gauge-num">{conf * 100:.0f}<tspan class="edn-gauge-unit">%</tspan>
      </text>
    </svg>
    <div class="edn-verdict">
      <div class="edn-kicker">Predicted class</div>
      <div class="edn-label">{escape(best)}</div>
      <div class="edn-sub">{conf * 100:.1f}% softmax confidence (not calibrated)</div>
    </div>
  </div>
  <ol class="edn-bars">{bars}</ol>
  <div class="edn-card">
    <p class="edn-card-title">About {escape(best)}</p>
    <p>{escape(info["summary"])}</p>
    <p><b>In this dataset:</b> {escape(info["looks_like"])}</p>
    <p class="edn-muted">{escape(info["source"])}</p>
  </div>
</div>"""


def analyse(image: Image.Image | None) -> str:
    """Gradio handler: image in, result-card HTML out."""
    return render_result(classify(image))


def _class_cards() -> str:
    cards = "".join(
        f'<article class="edn-class" style="--c:{CLASS_INFO[name]["color"]}"><h4>{escape(name)}</h4>'
        f"<p>{escape(CLASS_INFO[name]['summary'])}</p><p class='edn-muted'>{escape(CLASS_INFO[name]['looks_like'])}</p></article>"
        for name in CLASS_NAMES
    )
    return f'<div class="edn-classes">{cards}</div>'


HEADER_HTML = """
<header class="edn-hero">
  <div class="edn-wafer" aria-hidden="true"><span></span></div>
  <div>
    <p class="edn-eyebrow">Wafer defect classifier</p>
    <h1>EdgeDefectNet</h1>
    <p class="edn-lede">A grayscale ResNet-18 sorts wafer images into nine defect classes. Upload an image or choose one of the
    held-out examples. For a version that runs entirely in your browser, open the
    <a href="https://sandeepsrinivasan-14.github.io/EdgeDefectNet/" target="_blank" rel="noopener">web app</a>.</p>
  </div>
</header>"""

CAVEAT_HTML = """
<p class="edn-note">Research demo, not for process decisions. Most training images are WM-811K-style wafer maps whose spatial
patterns were mapped onto the class names, so those names are proxy labels. The model always answers with one of the nine
classes, even for images that are not wafers. See the
<a href="https://github.com/Sandeepsrinivasan-14/EdgeDefectNet/blob/main/MODEL_CARD.md"
target="_blank" rel="noopener">model card</a>.</p>"""

CSS = (ROOT / "app.css").read_text(encoding="utf-8")


def build_theme() -> Any:
    import gradio as gr

    return gr.themes.Base(
        primary_hue="violet",
        secondary_hue="cyan",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_lg,
        font=["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],  # system fonts: nothing to download
    )


def build_demo() -> Any:
    import gradio as gr

    example_files = sorted(EXAMPLES_DIR.glob("*.png"))
    with gr.Blocks(title="EdgeDefectNet") as demo:
        gr.HTML(HEADER_HTML)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="edn-glass"):
                image = gr.Image(
                    type="pil",
                    image_mode="L",
                    label="Wafer image",
                    sources=["upload", "clipboard"],
                    height=340,
                )
                if example_files:
                    gr.Examples(
                        examples=[[str(p)] for p in example_files],
                        inputs=image,
                        label="Held-out examples (one per class)",
                        examples_per_page=9,
                    )
            with gr.Column(scale=6, elem_classes="edn-glass"):
                result = gr.HTML(render_result({}), label="Prediction")
        gr.HTML(CAVEAT_HTML)
        with gr.Accordion("The nine classes", open=False, elem_classes="edn-glass"):
            gr.HTML(_class_cards())
        image.change(analyse, inputs=image, outputs=result, show_progress="minimal", api_name="classify")
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    get_predictor()  # fail fast at start-up if the model is missing or corrupt
    build_demo().launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        theme=build_theme(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
