## What and why

<!-- One or two sentences. Link the issue if there is one: Fixes #123 -->

## Checklist

- [ ] `ruff check .`, `ruff format --check .` and `mypy` pass
- [ ] `pytest` passes; new behaviour has tests
- [ ] If the model was retrained: `models/edgedefectnet.onnx.sha256`, `reports/`, `MODEL_CARD.md` and the README metrics are updated together
- [ ] `CHANGELOG.md` updated for user-visible changes
- [ ] No datasets, checkpoints or other large binaries added (see CONTRIBUTING.md)
