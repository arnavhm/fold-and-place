"""Read the downloaded datasets (data/indofashion, data/myntra) into (image path, category) samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fold_and_place.vision.labels import map_indofashion, map_myntra


@dataclass(frozen=True)
class Sample:
    path: Path
    category: str
    source: str  # indofashion | myntra
    split: str  # train | test


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find(root: Path, name: str) -> Path | None:
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def indofashion(root: Path) -> list[Sample]:
    out = []
    for split, fname in (("train", "train_data.json"), ("test", "test_data.json")):
        f = _find(root, fname)
        if f is None:
            continue
        base = f.parent
        for r in _jsonl(f):
            cat = map_indofashion(r.get("class_label", ""))
            p = base / r.get("image_path", "")
            if cat and p.exists():
                out.append(Sample(p, cat, "indofashion", split))
    return out


def myntra(root: Path, test_frac: float = 0.25, seed: int = 0) -> list[Sample]:
    import pandas as pd

    styles = _find(root, "styles.csv")
    if styles is None:
        return []
    df = pd.read_csv(styles, on_bad_lines="skip")
    img_dir = styles.parent / "images"
    rng = np.random.default_rng(seed)
    out = []
    for _, r in df.iterrows():
        cat = map_myntra(r.get("articleType", ""), r.get("gender", ""))
        p = img_dir / f"{r['id']}.jpg"
        if cat and p.exists():
            out.append(Sample(p, cat, "myntra", "test" if rng.random() < test_frac else "train"))
    return out


def sample_balanced(samples: list[Sample], n_train: int, n_test: int, seed: int = 0) -> list[Sample]:
    """At most n_train / n_test images per (category, source)."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[Sample]] = {}
    for s in samples:
        groups.setdefault((s.category, s.source, s.split), []).append(s)
    out = []
    for (_, _, split), items in sorted(groups.items(), key=lambda kv: kv[0]):
        k = n_train if split == "train" else n_test
        idx = rng.permutation(len(items))[:k]
        out += [items[i] for i in idx]
    return out
