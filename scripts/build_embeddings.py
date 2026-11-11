"""Embed a balanced sample of the downloaded datasets, train the classifier head, run E2.

    python scripts/build_embeddings.py                 # 150 train + 50 test per category/source
    python scripts/build_embeddings.py --train 60 --test 20   # quicker

Writes data/embeddings.npz (not committed), models/head.npz (not committed) and results/E2_*.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fold_and_place.vision import datasets  # noqa: E402
from fold_and_place.vision.labels import category_prompts  # noqa: E402
from fold_and_place.vision.models import Embedder, LinearHead  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--test", type=int, default=50)
    args = ap.parse_args()
    data = ROOT / "data"
    samples = datasets.indofashion(data / "indofashion") + datasets.myntra(data / "myntra")
    if not samples:
        sys.exit("No images found under data/. Run `make data` first.")
    picked = datasets.sample_balanced(samples, args.train, args.test)
    print(f"{len(samples)} labelled images found; embedding {len(picked)}")

    from PIL import Image

    emb = Embedder()
    t = time.time()
    X, keep = [], []
    batch = 64
    for i in range(0, len(picked), batch):
        chunk = picked[i : i + batch]
        imgs, ok = [], []
        for s in chunk:
            try:
                imgs.append(Image.open(s.path).convert("RGB"))
                ok.append(s)
            except Exception:  # noqa: BLE001 - skip unreadable files
                continue
        if imgs:
            X.append(emb.images(imgs))
            keep += ok
        print(f"  {min(i + batch, len(picked))}/{len(picked)}  ({time.time() - t:.0f}s)", end="\r")
    X = np.concatenate(X)
    print()
    labels, bank = emb.prompt_bank(category_prompts())
    y = np.array([s.category for s in keep])
    split = np.array([s.split for s in keep])
    source = np.array([s.source for s in keep])
    data.mkdir(exist_ok=True)
    np.savez(
        data / "embeddings.npz",
        X=X,
        y=y,
        split=split,
        source=source,
        paths=np.array([str(s.path) for s in keep]),
        label_names=np.array(labels),
        label_bank=bank,
    )
    head = LinearHead.train(X[split == "train"], list(y[split == "train"]))
    (ROOT / "models").mkdir(exist_ok=True)
    head.save(ROOT / "models" / "head.npz")
    print(f"head trained on {int((split == 'train').sum())} images, {len(head.classes)} classes")

    from fold_and_place.experiments import e2_vision

    print(e2_vision.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
