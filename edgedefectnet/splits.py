"""Leakage-free dataset splitting.

The largest source is a Roboflow export in which each original wafer map appears several times
as augmented copies (filenames share a stem and differ only in the ".rf.<hash>" suffix). Version 1
split at the file level, so a third of the test images had a sibling of the same wafer in the
training set. Here every copy of an original wafer is assigned to the same split.
"""

import csv
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Item:
    filename: str  # e.g. MULTI_CLASS__image_Center_12116_png_jpg.rf.3b0b....png
    label: str
    source: str
    group: str
    split: str = ""


def group_id(filename: str) -> tuple[str, str]:
    """Return (source, group) for a processed filename like 'SOURCE__originalstem[.rf.hash].png'."""
    stem = Path(filename).stem
    source, sep, rest = stem.partition("__")
    if not sep:
        source, rest = "UNKNOWN", stem
    return source, f"{source}:{RF_SUFFIX.sub('', rest)}"


def assign_splits(items: list[Item], fractions=(0.70, 0.15, 0.15), seed: int = 42) -> list[Item]:
    """Stratified (per class) group split: whole groups go to one split, sized by image count."""
    by_class: dict[str, dict[str, list[Item]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_class[it.label][it.group].append(it)

    rng = random.Random(seed)
    out: list[Item] = []
    for label in sorted(by_class):
        groups = sorted(by_class[label])
        rng.shuffle(groups)
        total = sum(len(by_class[label][g]) for g in groups)
        targets = [f * total for f in fractions]
        counts = [0, 0, 0]
        for g in groups:
            members = by_class[label][g]
            # put the group in the split that is furthest below its target
            idx = max(range(3), key=lambda i: (targets[i] - counts[i]) / max(targets[i], 1e-9))
            counts[idx] += len(members)
            out.extend(Item(m.filename, m.label, m.source, m.group, SPLITS[idx]) for m in members)
    return sorted(out, key=lambda i: (i.split, i.label, i.filename))


def write_manifest(items: list[Item], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "label", "source", "group", "filename"])
        for it in items:
            writer.writerow([it.split, it.label, it.source, it.group, it.filename])


def read_manifest(path: str | Path) -> list[Item]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [Item(r["filename"], r["label"], r["source"], r["group"], r["split"]) for r in csv.DictReader(fh)]
