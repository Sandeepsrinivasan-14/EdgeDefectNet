"""Image preprocessing shared by training, evaluation and inference.

Kept in NumPy/Pillow (no torch) so the ONNX inference path has no heavy dependencies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = 224
NORM_MEAN = 0.45  # dataset grayscale mean
NORM_STD = 0.23  # dataset grayscale std

#: Refuse files larger than this before decoding (a wafer map is a few KB to a few MB).
MAX_FILE_BYTES = 25 * 1024 * 1024
#: Refuse images with more pixels than this (guards against decompression bombs).
MAX_PIXELS = 64_000_000
#: Images smaller than this on either side carry no usable signal.
MIN_SIDE = 8


class InvalidImageError(ValueError):
    """The input could not be read as a usable image."""


def validate_image(img: Image.Image) -> Image.Image:
    """Check an already-decoded image's dimensions; returns it unchanged."""
    width, height = img.size
    if min(width, height) < MIN_SIDE:
        raise InvalidImageError(f"image is too small ({width}x{height}); minimum side is {MIN_SIDE}px")
    if width * height > MAX_PIXELS:
        raise InvalidImageError(f"image is too large ({width}x{height}); maximum is {MAX_PIXELS:,} pixels")
    return img


def load_grayscale(path: str | Path) -> Image.Image:
    """Read an image file as an 8-bit grayscale PIL image, raising InvalidImageError on any problem."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise InvalidImageError(f"image file is empty: {path}")
    if size > MAX_FILE_BYTES:
        raise InvalidImageError(f"image file is too large ({size / 1e6:.1f} MB; limit {MAX_FILE_BYTES / 1e6:.0f} MB): {path}")
    try:
        with Image.open(path) as img:
            validate_image(img)  # dimensions come from the header, before the pixels are decoded
            return img.convert("L")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"cannot read {path} as an image: {exc}") from exc


def to_input_array(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 array of shape (1, 1, 224, 224), normalised like the training data."""
    validate_image(img)
    img = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - NORM_MEAN) / NORM_STD
    return arr[None, None, :, :]
