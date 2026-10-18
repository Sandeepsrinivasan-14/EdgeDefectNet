"""Input validation and error handling for the inference path."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edgedefectnet import predict as predict_mod
from edgedefectnet.classes import CLASS_NAMES
from edgedefectnet.predict import ModelError, Predictor, expected_sha256, sha256_file, verify_model
from edgedefectnet.preprocessing import (
    MAX_PIXELS,
    InvalidImageError,
    load_grayscale,
    resolve_image_path,
    to_input_array,
    validate_image,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "edgedefectnet.onnx"
EXAMPLE = sorted((ROOT / "examples").glob("*.png"))[0]


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(MODEL)


def test_missing_image_file(predictor, tmp_path):
    with pytest.raises(FileNotFoundError):
        predictor.predict(tmp_path / "missing.png")


def test_empty_image_file(predictor, tmp_path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(InvalidImageError, match="empty"):
        predictor.predict(empty)


def test_non_image_file(predictor, tmp_path):
    bad = tmp_path / "notes.png"
    bad.write_text("this is not a PNG")
    with pytest.raises(InvalidImageError, match="cannot read"):
        predictor.predict(bad)


def test_truncated_image_file(predictor, tmp_path):
    truncated = tmp_path / "cut.png"
    truncated.write_bytes(EXAMPLE.read_bytes()[:100])
    with pytest.raises(InvalidImageError):
        predictor.predict(truncated)


def test_directory_is_not_an_image(predictor, tmp_path):
    with pytest.raises(FileNotFoundError):
        predictor.predict(tmp_path)


def test_too_small_image(predictor):
    with pytest.raises(InvalidImageError, match="too small"):
        predictor.predict(Image.new("L", (3, 3)))


def test_too_many_pixels_rejected_without_allocating():
    class Huge:
        size = (MAX_PIXELS // 1000, 1001)

    with pytest.raises(InvalidImageError, match="too large"):
        validate_image(Huge())  # type: ignore[arg-type]


def test_oversized_file_rejected(monkeypatch):
    import edgedefectnet.preprocessing as pre

    monkeypatch.setattr(pre, "MAX_FILE_BYTES", 10)
    with pytest.raises(InvalidImageError, match="too large"):
        load_grayscale(EXAMPLE)


@pytest.mark.parametrize("bad", [None, 42, np.zeros((224, 224)), b"bytes"])
def test_wrong_input_type(predictor, bad):
    with pytest.raises(TypeError):
        predictor.predict(bad)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P", "I;16", "1"])
def test_all_common_pillow_modes_are_accepted(predictor, mode):
    img = Image.new(mode, (40, 30))
    assert to_input_array(img).shape == (1, 1, 224, 224)
    assert predictor.predict(img).label in CLASS_NAMES


def test_prediction_is_deterministic_and_normalised(predictor):
    a, b = predictor.predict(EXAMPLE), predictor.predict(EXAMPLE)
    assert a == b
    assert sum(a.probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_model_path_accepts_str_and_env_var(monkeypatch):
    assert Predictor(str(MODEL)).model_path == MODEL
    monkeypatch.setenv(predict_mod.MODEL_ENV_VAR, str(MODEL))
    assert Predictor().model_path == MODEL


def test_env_var_pointing_nowhere_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv(predict_mod.MODEL_ENV_VAR, str(tmp_path / "nope.onnx"))
    with pytest.raises(FileNotFoundError, match="EDGEDEFECTNET_MODEL"):
        Predictor()


def test_corrupt_model_file(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"\x00" * 128)
    with pytest.raises(ModelError, match="cannot load"):
        Predictor(bad)


def test_empty_model_file(tmp_path):
    empty = tmp_path / "empty.onnx"
    empty.write_bytes(b"")
    with pytest.raises(ModelError, match="empty"):
        Predictor(empty)


def test_git_lfs_pointer_is_diagnosed(tmp_path):
    pointer = tmp_path / "model.onnx"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 44689933\n")
    with pytest.raises(ModelError, match="Git LFS pointer"):
        Predictor(pointer)


def test_committed_checksum_matches_model():
    """models/edgedefectnet.onnx.sha256 must be updated whenever the model is retrained."""
    expected = expected_sha256(MODEL)
    assert expected == sha256_file(MODEL)
    assert verify_model(MODEL) == expected
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == expected
    Predictor(MODEL, verify=True)


def test_verify_detects_tampering(tmp_path):
    copy = tmp_path / "m.onnx"
    copy.write_bytes(MODEL.read_bytes() + b"\x00")
    (tmp_path / "m.onnx.sha256").write_text(f"{expected_sha256(MODEL)}  m.onnx\n")
    with pytest.raises(ModelError, match="mismatch"):
        Predictor(copy, verify=True)


def test_verify_without_sidecar_fails_closed(tmp_path):
    copy = tmp_path / "m.onnx"
    copy.write_bytes(MODEL.read_bytes())
    with pytest.raises(ModelError, match="no expected SHA-256"):
        Predictor(copy, verify=True)


def test_cli_reports_bad_images_and_keeps_going(capsys, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("nope")
    code = predict_mod.main([str(bad), str(EXAMPLE), "--json", "--model", str(MODEL)])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 1
    assert "error" in rows[0]
    assert rows[1]["label"] in CLASS_NAMES


def test_cli_text_output_and_success_exit_code(capsys):
    assert predict_mod.main([str(EXAMPLE), "--top", "2"]) == 0
    assert EXAMPLE.name in capsys.readouterr().out


def test_cli_missing_model_exits_2(capsys, tmp_path):
    assert predict_mod.main([str(EXAMPLE), "--model", str(tmp_path / "nope.onnx")]) == 2
    assert "model not found" in capsys.readouterr().err


@pytest.mark.parametrize("top", ["0", "10", "-1"])
def test_cli_rejects_bad_top(top):
    with pytest.raises(SystemExit) as exc:
        predict_mod.main([str(EXAMPLE), "--top", top])
    assert exc.value.code == 2


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        predict_mod.main(["--version"])
    assert exc.value.code == 0
    assert "edgedefectnet" in capsys.readouterr().out


# ---- path confinement (allowed_root) ----


@pytest.fixture
def jail(tmp_path):
    """A directory with one image inside it and one secret image next to it."""
    root = tmp_path / "uploads"
    root.mkdir()
    (root / "ok.png").write_bytes(EXAMPLE.read_bytes())
    (tmp_path / "secret.png").write_bytes(EXAMPLE.read_bytes())
    return root


def test_relative_and_absolute_paths_inside_root_are_accepted(jail):
    assert resolve_image_path("ok.png", jail) == (jail / "ok.png").resolve()
    assert resolve_image_path(jail / "ok.png", jail) == (jail / "ok.png").resolve()
    assert load_grayscale("ok.png", jail).mode == "L"


@pytest.mark.parametrize("evil", ["../secret.png", "sub/../../secret.png", "./../secret.png"])
def test_dot_dot_traversal_is_rejected(jail, evil):
    with pytest.raises(InvalidImageError, match="outside the allowed directory"):
        resolve_image_path(evil, jail)


def test_absolute_path_outside_root_is_rejected(jail):
    with pytest.raises(InvalidImageError, match="outside the allowed directory"):
        resolve_image_path(jail.parent / "secret.png", jail)


def test_sibling_directory_with_common_prefix_is_rejected(jail):
    sibling = jail.parent / "uploads-evil"
    sibling.mkdir()
    (sibling / "x.png").write_bytes(EXAMPLE.read_bytes())
    with pytest.raises(InvalidImageError, match="outside the allowed directory"):
        resolve_image_path(sibling / "x.png", jail)


def test_symlink_escaping_the_root_is_rejected(jail):
    link = jail / "link.png"
    try:
        link.symlink_to(jail.parent / "secret.png")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform or account")
    with pytest.raises(InvalidImageError, match="outside the allowed directory"):
        resolve_image_path("link.png", jail)


def test_symlink_inside_the_root_is_followed(jail):
    link = jail / "alias.png"
    try:
        link.symlink_to(jail / "ok.png")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform or account")
    assert resolve_image_path("alias.png", jail) == (jail / "ok.png").resolve()


def test_nul_byte_in_path_is_rejected(jail):
    with pytest.raises(InvalidImageError, match="NUL"):
        resolve_image_path("ok.png" + chr(0) + ".txt", jail)


def test_missing_file_and_directories_inside_root(jail):
    (jail / "dir").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_image_path("missing.png", jail)
    with pytest.raises(FileNotFoundError):
        resolve_image_path("dir", jail)


def test_predictor_enforces_allowed_root(jail):
    confined = Predictor(MODEL, allowed_root=jail)
    assert confined.predict("ok.png").label in CLASS_NAMES
    with pytest.raises(InvalidImageError, match="outside"):
        confined.predict("../secret.png")


def test_predict_image_takes_pil_only(predictor):
    assert predictor.predict_image(Image.open(EXAMPLE)).label == predictor.predict(EXAMPLE).label
    with pytest.raises(TypeError):
        predictor.predict_image(str(EXAMPLE))  # type: ignore[arg-type]


def test_cli_root_flag_confines_paths(capsys, jail):
    code = predict_mod.main([str(jail.parent / "secret.png"), "--root", str(jail), "--json"])
    assert code == 1
    assert "outside the allowed directory" in capsys.readouterr().out
    assert predict_mod.main(["ok.png", "--root", str(jail)]) == 0
