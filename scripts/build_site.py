"""Assemble the static web app into _site/ (used by the GitHub Pages workflow and the browser tests).

python scripts/build_site.py            # then: python -m http.server -d _site 8000

The site is web/ plus the model, the held-out example images, the test metrics and classes.json (class names, colours
and descriptions generated from edgedefectnet/classes.py, so the browser and the Python package share one source).
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets this script run without `pip install -e .` (the Pages workflow does not install)

from edgedefectnet.classes import CLASS_INFO, CLASS_NAMES  # noqa: E402


def class_manifest() -> list[dict[str, str]]:
    """The classes in logit order, each with its display metadata."""
    return [
        {
            "name": name,
            "color": CLASS_INFO[name]["color"],
            "summary": CLASS_INFO[name]["summary"],
            "looks_like": CLASS_INFO[name]["looks_like"],
            "source": CLASS_INFO[name]["source"],
        }
        for name in CLASS_NAMES
    ]


def build(out: Path = ROOT / "_site") -> Path:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out)
    shutil.copy2(ROOT / "models" / "edgedefectnet.onnx", out / "model.onnx")
    shutil.copy2(ROOT / "reports" / "metrics_test.json", out / "metrics.json")
    (out / "classes.json").write_text(json.dumps(class_manifest(), indent=1), encoding="utf-8")
    examples = out / "examples"
    examples.mkdir()
    files = sorted(p.name for p in (ROOT / "examples").glob("*.png"))
    for name in files:
        shutil.copy2(ROOT / "examples" / name, examples / name)
    (examples / "index.json").write_text(json.dumps(files))
    (out / ".nojekyll").write_text("")
    return out


if __name__ == "__main__":
    print(f"site written to {build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / '_site')}")
