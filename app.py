"""Gradio demo: upload a wafer image and get the predicted class with probabilities.

pip install -r requirements.txt gradio
python app.py            # http://127.0.0.1:7860
"""

from pathlib import Path

import gradio as gr

from edgedefectnet.predict import Predictor

ROOT = Path(__file__).resolve().parent
predictor = Predictor(ROOT / "models" / "edgedefectnet.onnx")


def classify(image):
    if image is None:
        return {}
    return predictor.predict(image).probabilities


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", image_mode="L", label="Wafer image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    examples=[[str(p)] for p in sorted((ROOT / "examples").glob("*.png"))],
    title="EdgeDefectNet",
    description="Grayscale ResNet-18 classifying wafer images into 9 defect classes. "
    "See the README for how the classes map to the source datasets.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
