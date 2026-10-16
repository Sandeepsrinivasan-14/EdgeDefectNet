# Contributing

Thanks for helping. Bug reports, fixes and documentation improvements are all welcome.

## Set up

```bash
git clone https://github.com/Sandeepsrinivasan-14/EdgeDefectNet
cd EdgeDefectNet
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,demo]"                          # add ",train" for PyTorch-side work
```

## Before you open a pull request

```bash
ruff check . && ruff format --check .     # lint and formatting (ruff.toml)
mypy                                       # types (configuration in pyproject.toml)
pytest --cov                               # tests + coverage gate for the inference package
```

CI runs the same commands on Linux, macOS and Windows with Python 3.10-3.13, plus a browser parity test
(`pip install playwright && playwright install chromium`, then `pytest tests/test_web.py`), the PyTorch tests, a
package build and a Docker build. Tests that need optional packages (PyTorch, Playwright, Gradio) skip when they
are missing.

Guidelines:

- Keep the inference path (`predict.py`, `preprocessing.py`, `classes.py`) free of PyTorch; it must run on
  `onnxruntime`, `numpy` and `pillow` alone.
- `CLASS_NAMES` is the single source of the class order. Do not introduce a second list (that was the v1 bug).
- Add a test for every behaviour change and a line to `CHANGELOG.md` for user-visible ones.
- Use clear commit messages; one logical change per commit.

## Changing the model

If you retrain or re-export `models/edgedefectnet.onnx`, update in the same pull request: the SHA-256 file
(`sha256sum models/edgedefectnet.onnx > models/edgedefectnet.onnx.sha256`), `reports/`, the numbers in `README.md`
and `MODEL_CARD.md`, and `CHANGELOG.md`. `tests/test_validation.py::test_committed_checksum_matches_model` fails
otherwise.

### How model files are stored, and why

The 44.7 MB ONNX file is a normal Git blob. We considered the alternatives:

| Option | Verdict |
|---|---|
| **Git LFS** | Not used. Source ZIP downloads, `pip install git+...`, the Docker build and Pages workflow all receive a tiny pointer file unless every consumer opts in to LFS; clones without `git-lfs` break silently (the loader detects and explains this case). GitHub's free LFS bandwidth quota (1 GB/month) is also easy to exhaust on a public repo with CI. |
| **GitHub Releases** | Preferred for future models. The release workflow (`.github/workflows/release.yml`) attaches the ONNX file and its SHA-256 to each tagged release; consumers download the asset and verify it, and the repository stays small. |
| **Keep it in Git (current)** | Acceptable only while the model is rarely replaced: it is below GitHub's 50 MB warning and 100 MB limit, and everything works from a plain clone. But **each retrained model adds about 42 MB to history permanently**. |

Recommendation: the next time the model changes, publish it as a Release asset, fetch it in the Docker/Pages
builds by URL plus SHA-256, and remove it from the tree rather than committing a third copy.

## Repository size

A fresh clone is large for a project this size (GitHub reports about 500 MB) because **history** contains data
that is no longer in the tree: about 15,000 dataset images (`Phase1_Dataset/`, `data/multi_class_wafer/`,
`IESA_final_dataset_preview/`) and the v1 model weights (`edge_defect_phase1.onnx.data`, 44.7 MB), all removed in
the v2 commit but still reachable from older commits. The current tree is about 48 MB, almost all of it the
model. Do not commit datasets, checkpoints or generated files (`.gitignore` already excludes `dataset/`,
`checkpoints/`, `*.pt`, `*.npz`, `*.zip`); the dataset is rebuilt from `data/manifest.csv` with
`scripts/build_dataset.py`.

Shrinking history requires rewriting it (for example `git filter-repo --invert-paths --path Phase1_Dataset
--path data/multi_class_wafer --path IESA_final_dataset_preview --path edge_defect_phase1.onnx.data`) followed by a
force-push, which changes every commit hash and invalidates existing clones and forks. That is a maintainer
decision and has deliberately not been done automatically.

## Reporting security issues

See [SECURITY.md](SECURITY.md). Do not file them as public issues.
