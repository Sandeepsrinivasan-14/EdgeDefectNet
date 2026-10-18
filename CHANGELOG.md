# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Redesigned browser app: glassmorphism UI with animated background, dark/light themes, drag-and-drop, paste and
  page-wide drop upload, animated radial confidence gauge and ranked bars, a low-confidence warning, per-class
  explainer cards with schematic wafer-map glyphs, and an interactive three.js wafer whose rim carries one arc per
  class (lazy-loaded, reduced-motion aware, 2D fallback). three.js and ONNX Runtime Web are pinned with SRI hashes.
- Redesigned Gradio demo in the same visual language (confidence gauge, bars, class explainers).
- `CLASS_INFO` (colour, summary and dataset notes per class) is the single source for both front ends; the site
  build emits it as `classes.json`.
- `resolve_image_path` / `allowed_root`: image paths can be confined to a directory (`..` and symlinks resolved
  before checking); `Predictor(allowed_root=...)`, `edgedefectnet-predict --root`, and `Predictor.predict_image`
  for in-memory images (no file-system access).
- Browser tests for upload, error handling and the theme toggle, and a check that the page raises no JS errors.
- `pyproject.toml` packaging (`pip install .`, `edgedefectnet-predict` console script, extras `demo`, `train`, `dev`).
- Input validation and clear errors for unreadable, empty, oversized or tiny images and for missing, corrupt or
  Git-LFS-pointer model files (`InvalidImageError`, `ModelError`).
- `EDGEDEFECTNET_MODEL` environment variable, `--verify` flag and `models/edgedefectnet.onnx.sha256` integrity check.
- Type hints throughout the inference path, `py.typed`, and a mypy configuration.
- Dockerfile for the Gradio demo, model card (`MODEL_CARD.md`), CONTRIBUTING, SECURITY, Dependabot, CodeQL and
  release workflows.
- CI: lint + type check, test matrix (Linux/macOS/Windows, Python 3.10-3.13) with coverage, browser parity test,
  package build and install smoke test, Docker build.

### Changed
- The Gradio demo now requires Gradio 6 (`gradio>=6,<7`) and classifies only in-memory images.
- The GitHub Pages workflow now checks that Pages is enabled and degrades to a build-only run with a clear notice
  instead of failing with a 404; it also builds on pull requests without deploying.
- The CLI continues past unreadable images and exits non-zero (1: bad image, 2: model could not be loaded).

### Security
- Resolved the two CodeQL `py/path-injection` alerts in `edgedefectnet/preprocessing.py`: the web demo no longer
  routes request data into path handling, and path inputs can be confined to an allowed root.

## [2.0.0]

- Leak-free split, corrected class names, retrained model and browser app (see `reports/v1_audit.md`).
