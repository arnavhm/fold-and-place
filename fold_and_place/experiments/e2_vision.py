"""E2 - Does vision recognise Indian and Western garments?

Needs data/embeddings.npz, made on your laptop by `make data` then `make embeddings`
(scripts/build_embeddings.py). Compares, on the held-out test images:
  zero-shot FashionSigLIP  vs  trained linear head  vs  their average (the pipeline's default).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fold_and_place.config import ROOT
from fold_and_place.experiments.common import out_dir, save_csv, save_fig, write_notes
from fold_and_place.vision.models import LinearHead, ensemble, zero_shot

EMB = ROOT / "data" / "embeddings.npz"
HEAD = ROOT / "models" / "head.npz"


def evaluate(X, y, split, source, label_names, label_bank, head: LinearHead) -> tuple[pd.DataFrame, dict]:
    te = split == "test"
    Xt, yt, st = X[te], y[te], source[te]
    zs = zero_shot(Xt, label_bank)
    hp = head.proba(Xt)
    preds = {
        "zero-shot": np.array(label_names)[zs.argmax(1)],
        "trained head": np.array(head.classes)[hp.argmax(1)],
        "ensemble": np.array(label_names)[ensemble(zs, label_names, head, hp).argmax(1)],
    }
    rows = []
    for name, p in preds.items():
        for src in ("all", "indofashion", "myntra"):
            m = np.ones(len(yt), bool) if src == "all" else (st == src)
            if m.sum():
                rows.append({"method": name, "subset": src, "n": int(m.sum()), "top1_accuracy": round(float((p[m] == yt[m]).mean()), 4)})
    return pd.DataFrame(rows), preds


def run(path=None) -> dict:
    d = out_dir(path)
    if not EMB.exists():
        write_notes(d, "E2_notes.md", "# E2 - not run yet\n\nRun `make data` and `make embeddings` on a machine with the datasets.")
        return {"status": "not run: data/embeddings.npz missing"}
    z = np.load(EMB, allow_pickle=False)
    X, y, split, source = z["X"], z["y"].astype(str), z["split"].astype(str), z["source"].astype(str)
    label_names = [str(s) for s in z["label_names"]]
    label_bank = z["label_bank"]
    head = LinearHead.load(HEAD) if HEAD.exists() else LinearHead.train(X[split == "train"], list(y[split == "train"]))
    acc, preds = evaluate(X, y, split, source, label_names, label_bank, head)
    save_csv(acc, d, "E2_accuracy.csv")

    yt = y[split == "test"]
    cats = sorted(set(yt))
    per = []
    for c in cats:
        m = yt == c
        per.append({"category": c, "n_test": int(m.sum()), **{k: round(float((v[m] == c).mean()), 3) for k, v in preds.items()}})
    per_df = pd.DataFrame(per)
    save_csv(per_df, d, "E2_per_category.csv")

    ens = preds["ensemble"]
    idx = {c: i for i, c in enumerate(cats)}
    cm = np.zeros((len(cats), len(cats)))
    for t, p in zip(yt, ens):
        if p in idx:
            cm[idx[t], idx[p]] += 1
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cats)), cats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cats)), cats, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("E2: confusion matrix, ensemble classifier (row-normalised)", fontsize=9)
    save_fig(fig, d, "E2_confusion.png")

    pivot = acc.pivot(index="method", columns="subset", values="top1_accuracy")
    summary = {k: pivot.loc[k].to_dict() for k in pivot.index}
    lines = "\n".join(f"| {m} | " + " | ".join(f"{100 * pivot.loc[m].get(s, float('nan')):.1f}%" for s in ("all", "indofashion", "myntra")) + " |" for m in pivot.index)
    worst = per_df.sort_values("ensemble").head(5)
    write_notes(
        d,
        "E2_notes.md",
        f"""
# E2 - Garment recognition

Held-out test images: {int((split == 'test').sum())} ({', '.join(f'{s} {int(((split == "test") & (source == s)).sum())}' for s in ('indofashion', 'myntra'))}),
{len(cats)} of our categories. Train images: {int((split == 'train').sum())}.

| Method | All | IndoFashion (Indian) | Myntra (Western + Indian) |
|---|---|---|---|
{lines}

Hardest categories (ensemble): {', '.join(f"{r.category} {100 * r.ensemble:.0f}%" for r in worst.itertuples())}.
Categories with no public training images (polo, lungi) rely on zero-shot only.
""",
    )
    return summary


if __name__ == "__main__":
    print(run())
