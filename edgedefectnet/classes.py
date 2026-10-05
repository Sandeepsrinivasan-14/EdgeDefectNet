"""Class definitions.

CLASS_NAMES is in alphabetical order because that is the label order torchvision's ImageFolder
assigns, and therefore the order of the model's output logits. Every component (training,
evaluation, ONNX inference, demo) must use this single list.

Version 1 of this project kept a second, non-alphabetical list and used it to name predictions,
so every prediction and the per-class report were mislabelled. tests/test_classes.py guards
against that regression.
"""

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

# How each source pattern was mapped onto the 9 challenge classes when the dataset was assembled.
# The public sources (WM-811K-derived wafer maps) describe wafer-level spatial failure patterns, not
# die-level electrical defects, so for those sources the class names below are proxy labels.
SOURCE_MAPPING: dict[str, dict[str, str]] = {
    "IESA": {
        "Cracks/Defect": "Cracks",
        "Vias/Defect": "Vias",
        "Vias/Clean": "Clean",
    },
    "WM-811K patterns (WM811K_IMG and MULTI_CLASS)": {
        "Center": "Shorts",
        "Donut": "Cracks",
        "Edge-Loc": "LER",
        "Edge-Ring": "LER",
        "Local": "Opens",
        "Near-Full": "Bridges",
        "None / Normal": "Clean",
        "Random": "Other",
        "Scratch": "Scratches",
    },
}
