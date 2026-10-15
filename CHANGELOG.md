# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
- The GitHub Pages workflow now checks that Pages is enabled and degrades to a build-only run with a clear notice
  instead of failing with a 404; it also builds on pull requests without deploying.
- The CLI continues past unreadable images and exits non-zero (1: bad image, 2: model could not be loaded).

## [2.0.0]

- Leak-free split, corrected class names, retrained model and browser app (see `reports/v1_audit.md`).
