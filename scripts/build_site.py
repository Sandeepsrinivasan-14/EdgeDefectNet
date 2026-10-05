"""Assemble the static web app into _site/ (used by the GitHub Pages workflow and the browser tests).

python scripts/build_site.py            # then: python -m http.server -d _site 8000
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(out: Path = ROOT / "_site") -> Path:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out)
    shutil.copy2(ROOT / "models" / "edgedefectnet.onnx", out / "model.onnx")
    shutil.copy2(ROOT / "reports" / "metrics_test.json", out / "metrics.json")
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
