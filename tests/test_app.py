"""The Gradio demo's request handler (skipped when gradio is not installed)."""

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from edgedefectnet.classes import CLASS_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("edn_app", ROOT / "app.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["edn_app"] = module
    spec.loader.exec_module(module)
    return module


def test_empty_upload_clears_output(app):
    assert app.classify(None) == {}


def test_classify_returns_probabilities(app):
    probs = app.classify(Image.open(next((ROOT / "examples").glob("Clean__*.png"))))
    assert set(probs) == set(CLASS_NAMES)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


def test_tiny_image_becomes_a_user_facing_error(app):
    with pytest.raises(gr.Error, match="too small"):
        app.classify(Image.new("L", (2, 2)))


def test_missing_model_becomes_a_user_facing_error(app, monkeypatch, tmp_path):
    app.get_predictor.cache_clear()
    monkeypatch.setenv("EDGEDEFECTNET_MODEL", str(tmp_path / "nope.onnx"))
    try:
        with pytest.raises(gr.Error, match="unavailable"):
            app.classify(Image.new("L", (64, 64)))
    finally:
        app.get_predictor.cache_clear()


def test_demo_builds(app):
    assert app.build_demo() is not None
