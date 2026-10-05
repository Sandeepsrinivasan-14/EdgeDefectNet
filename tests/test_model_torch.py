"""Training-side checks; skipped when PyTorch is not installed (the inference package doesn't need it)."""

import pytest

torch = pytest.importorskip("torch")

from edgedefectnet.classes import NUM_CLASSES  # noqa: E402
from edgedefectnet.model import build_model  # noqa: E402


def test_grayscale_resnet_shapes():
    model = build_model().eval()
    assert model.conv1.in_channels == 1
    with torch.no_grad():
        out = model(torch.randn(2, 1, 224, 224))
    assert out.shape == (2, NUM_CLASSES)


def test_onnx_export_matches_torch(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from edgedefectnet.export import export

    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": build_model().state_dict()}, ckpt)
    diff = export(ckpt, tmp_path / "m.onnx")
    assert diff < 1e-3
