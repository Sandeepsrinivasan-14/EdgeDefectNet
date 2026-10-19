"""Class definitions.

CLASS_NAMES is in alphabetical order because that is the label order torchvision's ImageFolder
assigns, and therefore the order of the model's output logits. Every component (training,
evaluation, ONNX inference, demo) must use this single list.

Version 1 of this project kept a second, non-alphabetical list and used it to name predictions,
so every prediction and the per-class report were mislabelled. tests/test_classes_and_splits.py
guards against that regression.
"""

from typing import TypedDict

CLASS_NAMES: list[str] = [
    "Bridges",
    "Clean",
    "Cracks",
    "LER",
    "Opens",
    "Other",
    "Scratches",
    "Shorts",
    "Vias",
]
NUM_CLASSES = len(CLASS_NAMES)


class ClassInfo(TypedDict):
    """Human-readable description of one class, shared by the web app and the Gradio demo."""

    color: str  # hex colour used for the class in both front ends (never the only cue: labels are always shown)
    summary: str  # what the defect is, in one sentence
    looks_like: str  # what the model actually sees in this dataset
    source: str  # where the training examples come from


CLASS_INFO: dict[str, ClassInfo] = {
    "Bridges": {
        "color": "#f472b6",
        "summary": "An unintended conductive link between neighbouring features that can short adjacent lines.",
        "looks_like": "Almost the whole wafer failing. In the WM-811K source this is the Near-Full pattern.",
        "source": "Proxy label from WM-811K-style wafer maps (Near-Full).",
    },
    "Clean": {
        "color": "#34d399",
        "summary": "No systematic defect: any failing dies are sparse and unstructured.",
        "looks_like": "A uniformly healthy wafer map with at most a few isolated failures.",
        "source": "WM-811K-style maps with no pattern, plus defect-free IESA images.",
    },
    "Cracks": {
        "color": "#f87171",
        "summary": "A fracture in the die or substrate, often caused by stress or handling.",
        "looks_like": "A ring-shaped band of failing dies (the WM-811K Donut pattern) or a visible fracture in IESA images.",
        "source": "Proxy label from WM-811K-style maps (Donut) plus real IESA crack images.",
    },
    "LER": {
        "color": "#fbbf24",
        "summary": "Line-edge roughness: ragged, uneven edges on printed lines that perturb device behaviour.",
        "looks_like": "Failing dies hugging the rim of the wafer, either as an arc or a complete ring.",
        "source": "Proxy label from WM-811K-style maps (Edge-Loc and Edge-Ring).",
    },
    "Opens": {
        "color": "#60a5fa",
        "summary": "A break in a conductor that leaves part of a circuit disconnected.",
        "looks_like": "A compact local cluster of failing dies in one region of the wafer.",
        "source": "Proxy label from WM-811K-style maps (Local).",
    },
    "Other": {
        "color": "#a1a1aa",
        "summary": "Failures that fit none of the named classes.",
        "looks_like": "Failing dies scattered across the wafer without a spatial structure.",
        "source": "Proxy label from WM-811K-style maps (Random).",
    },
    "Scratches": {
        "color": "#fb923c",
        "summary": "Mechanical damage from handling or polishing that leaves a thin line of dead dies.",
        "looks_like": "A narrow, elongated streak of failing dies, usually curved or straight across the wafer.",
        "source": "Proxy label from WM-811K-style maps (Scratch).",
    },
    "Shorts": {
        "color": "#a78bfa",
        "summary": "An unintended connection between two nets that should be isolated.",
        "looks_like": "Failing dies concentrated at the centre of the wafer (the WM-811K Center pattern).",
        "source": "Proxy label from WM-811K-style maps (Center).",
    },
    "Vias": {
        "color": "#22d3ee",
        "summary": "A defective vertical interconnect between metal layers.",
        "looks_like": "Real via defects from the IESA challenge. Only 23 images exist in total, so treat it with caution.",
        "source": "IESA challenge images (the only real via examples).",
    },
}
