"""Image preprocessing shared by training, evaluation and inference.

Kept in NumPy/Pillow (no torch) so the ONNX inference path has no heavy dependencies.
"""

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 224
NORM_MEAN = 0.45  # dataset grayscale mean
NORM_STD = 0.23  # dataset grayscale std


def load_grayscale(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("L")


def to_input_array(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 array of shape (1, 1, 224, 224), normalised like the training data."""
    img = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - NORM_MEAN) / NORM_STD
    return arr[None, None, :, :]
