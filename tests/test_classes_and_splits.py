from collections import defaultdict
from pathlib import Path

from edgedefectnet.classes import CLASS_INFO, CLASS_NAMES
from edgedefectnet.splits import Item, assign_splits, group_id, read_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_class_names_are_in_imagefolder_order():
    # torchvision.datasets.ImageFolder assigns labels in sorted folder order; the model's logits follow it.
    assert sorted(CLASS_NAMES) == CLASS_NAMES
    assert len(CLASS_NAMES) == 9


def test_every_class_is_described_with_a_distinct_colour():
    assert list(CLASS_INFO) == CLASS_NAMES
    colours = [info["color"] for info in CLASS_INFO.values()]
    assert len(set(colours)) == len(CLASS_NAMES)
    assert all(info["summary"] and info["looks_like"] and info["source"] for info in CLASS_INFO.values())


def test_group_id_strips_roboflow_suffix():
    a = group_id("MULTI_CLASS__image_Center_12116_png_jpg.rf.3b0bff20d31a188c6d049bb8bc4eb385.png")
    b = group_id("MULTI_CLASS__image_Center_12116_png_jpg.rf.625759f574d2c4bceb7ed3ac017047ea.png")
    assert a == b == ("MULTI_CLASS", "MULTI_CLASS:image_Center_12116_png_jpg")
    assert group_id("IESA__defect_0001.png") == ("IESA", "IESA:defect_0001")


def test_assign_splits_keeps_groups_together_and_is_deterministic():
    items = [Item(f"S__img{g}_png.rf.{k}.png", "A" if g % 2 else "B", "S", f"S:img{g}") for g in range(200) for k in range(3)]
    first = assign_splits(items, seed=1)
    assert first == assign_splits(items, seed=1)
    splits_per_group = defaultdict(set)
    for it in first:
        splits_per_group[it.group].add(it.split)
    assert all(len(s) == 1 for s in splits_per_group.values())
    counts = {s: sum(it.split == s for it in first) for s in ("train", "val", "test")}
    assert 0.65 < counts["train"] / len(items) < 0.75


def test_published_manifest_has_no_leakage():
    items = read_manifest(ROOT / "data" / "manifest.csv")
    assert len(items) > 7000
    by_group = defaultdict(set)
    for it in items:
        by_group[it.group].add(it.split)
        assert it.label in CLASS_NAMES
    leaking = [g for g, s in by_group.items() if len(s) > 1]
    assert leaking == []
    assert {it.split for it in items} == {"train", "val", "test"}
