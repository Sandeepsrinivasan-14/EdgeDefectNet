"""The static site assembled for GitHub Pages contains everything the web app fetches."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_site_contents(tmp_path):
    site = _load_builder().build(tmp_path / "_site")
    assert (site / "index.html").is_file()
    assert (site / "model.onnx").stat().st_size == (ROOT / "models" / "edgedefectnet.onnx").stat().st_size
    assert json.loads((site / "metrics.json").read_text())["n"] > 0
    listed = json.loads((site / "examples" / "index.json").read_text())
    assert listed
    assert all((site / "examples" / name).is_file() for name in listed)
    assert (site / ".nojekyll").exists()


def test_site_rebuild_is_idempotent(tmp_path):
    builder = _load_builder()
    builder.build(tmp_path / "_site")
    builder.build(tmp_path / "_site")
