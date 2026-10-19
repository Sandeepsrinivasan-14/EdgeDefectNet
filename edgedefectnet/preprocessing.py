"""Image preprocessing shared by training, evaluation and inference.

Kept in NumPy/Pillow (no torch) so the ONNX inference path has no heavy dependencies.
"""

from __future__ import annotations

import os
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


def resolve_image_path(path: str | os.PathLike[str], allowed_root: str | os.PathLike[str] | None = None) -> Path:
    """Return the canonical, absolute location of an image file, optionally confined to ``allowed_root``.

    The path is fully resolved (``..`` segments removed, symbolic links followed) *before* it is checked, so neither
    ``../`` traversal nor a symlink pointing outside the root can escape. A relative ``path`` is taken relative to
    ``allowed_root``; an absolute one must already lie inside it. Without ``allowed_root`` the path is only
    canonicalised, which is right for a command-line tool acting on the invoking user's own files; code that
    handles paths from other parties (a web service, an upload directory) should always pass a root.

    Raises:
        InvalidImageError: the path contains a NUL byte or resolves outside ``allowed_root``.
        FileNotFoundError: the path does not resolve to a regular file.
    """
    raw = os.fspath(path)
    if "\x00" in raw:
        raise InvalidImageError("image path contains a NUL byte")
    if allowed_root is None:
        resolved = os.path.realpath(raw)
    else:
        root = os.path.realpath(allowed_root)
        resolved = os.path.realpath(os.path.join(root, raw))  # an absolute `raw` replaces `root`; checked below
        if resolved != root and not resolved.startswith(root.rstrip(os.sep) + os.sep):
            raise InvalidImageError(f"image path is outside the allowed directory: {raw}")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"image not found: {raw}")
    return Path(resolved)


def load_grayscale(path: str | os.PathLike[str], allowed_root: str | os.PathLike[str] | None = None) -> Image.Image:
    """Read an image file as an 8-bit grayscale PIL image, raising InvalidImageError on any problem.

    ``allowed_root`` confines the lookup to one directory tree; see :func:`resolve_image_path`.
    """
    resolved = resolve_image_path(path, allowed_root)
    size = resolved.stat().st_size
    if size == 0:
        raise InvalidImageError(f"image file is empty: {resolved.name}")
    if size > MAX_FILE_BYTES:
        raise InvalidImageError(
            f"image file is too large ({size / 1e6:.1f} MB; limit {MAX_FILE_BYTES / 1e6:.0f} MB): {resolved.name}"
        )
    try:
        with Image.open(resolved) as img:
            validate_image(img)  # dimensions come from the header, before the pixels are decoded
            gray: Image.Image = img.convert("L")
            return gray
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"cannot read {resolved.name} as an image: {exc}") from exc


def to_input_array(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 array of shape (1, 1, 224, 224), normalised like the training data."""
    validate_image(img)
    img = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - NORM_MEAN) / NORM_STD
    return arr[None, None, :, :]
