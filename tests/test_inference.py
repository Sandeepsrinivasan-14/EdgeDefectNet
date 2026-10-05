import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edgedefectnet.classes import CLASS_NAMES
from edgedefectnet.predict import Predictor, softmax
from edgedefectnet.preprocessing import IMAGE_SIZE, to_input_array

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "edgedefectnet.onnx"
EXAMPLES = sorted((ROOT / "examples").glob("*.png"))


def test_preprocessing_shape_and_normalisation():
    arr = to_input_array(Image.new("L", (64, 32), color=115))  # 115/255 ~= 0.451 ~= mean
    assert arr.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert arr.dtype == np.float32
    assert abs(float(arr.mean())) < 0.01
    rgb = to_input_array(Image.new("RGB", (300, 300), color=(255, 255, 255)))
    assert rgb.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE)


def test_softmax():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(p.sum(), 1) and p.argmax() == 2


@pytest.fixture(scope="module")
def predictor():
    return Predictor(MODEL)


def test_model_outputs_one_logit_per_class(predictor):
    assert predictor.logits(Image.new("L", (224, 224))).shape == (len(CLASS_NAMES),)


def test_examples_cover_every_class():
    labels = {p.name.split("__")[0] for p in EXAMPLES}
    assert labels == set(CLASS_NAMES)


def test_predictions_on_examples_are_valid(predictor):
    correct = 0
    for path in EXAMPLES:
        pred = predictor.predict(path)
        assert pred.label in CLASS_NAMES
        assert np.isclose(sum(pred.probabilities.values()), 1, atol=1e-4)
        correct += pred.label == path.name.split("__")[0]
    # one held-out example per class; the model should get the large majority right
    assert correct >= len(EXAMPLES) - 2


def test_cli_json(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "edgedefectnet.predict", str(EXAMPLES[0]), "--json"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    row = json.loads(out.stdout.strip())
    assert row["label"] in CLASS_NAMES and 0 <= row["confidence"] <= 1


def test_missing_model_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path / "nope.onnx")
