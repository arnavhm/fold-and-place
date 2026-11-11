"""Photo -> garment record.

1. photo type (flat-lay / worn / folded), zero-shot
2. mask: rembg for flat-lay, SegFormer clothes parser for worn photos, plain-background fallback
3. category: zero-shot FashionSigLIP, averaged with the trained head when one exists
4. fabric and embellishment, zero-shot
5. size: marker -> size chart -> fabric standard (see measure.py), with the source recorded
Every component is injectable, so tests run with fakes and no model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fold_and_place import config
from fold_and_place.garments.record import Garment, make_garment
from fold_and_place.vision import labels as L
from fold_and_place.vision.measure import measure_with_marker, size_from_chart
from fold_and_place.vision.models import (
    Embedder,
    FlatSegmenter,
    LinearHead,
    PlainSegmenter,
    WornSegmenter,
    ensemble,
    zero_shot,
)

HEAD_PATH = config.ROOT / "models" / "head.npz"
SIZE_CONFIDENCE = {"marker": 0.9, "fabric_standard": 0.7, "size_chart": 0.6}


@dataclass
class Guess:
    category: str
    top3: list[tuple[str, float]]
    photo_type: str
    fabric: str
    embellish: str
    flat_w: float
    flat_l: float
    size_source: str
    confidence: float
    sleeve_len: float | None = None
    mask: np.ndarray | None = field(default=None, repr=False)

    def to_garment(self, gid: str, owner: str = "", value: float = 5.0, wear_freq: float = 1.0, size: str = "M") -> Garment:
        g = make_garment(
            self.category,
            gid,
            size,
            owner=owner,
            value=value,
            wear_freq=wear_freq,
            fabric=self.fabric,
            embellish=self.embellish,
            flat_w=self.flat_w,
            flat_l=self.flat_l,
            size_source=self.size_source,
            confidence=self.confidence,
        )
        if self.sleeve_len is not None and g.template in ("tee", "dress"):
            g.sleeve_len = self.sleeve_len
        return g


class Analyzer:
    def __init__(self, embedder=None, flat_seg=None, worn_seg=None, head: LinearHead | None = None, use_head: bool = True):
        self.embedder = embedder or Embedder()
        self.flat_seg = flat_seg
        self.worn_seg = worn_seg
        self.fallback_seg = PlainSegmenter()
        if head is None and use_head and Path(HEAD_PATH).exists():
            head = LinearHead.load(HEAD_PATH)
        self.head = head
        self._banks: dict[str, tuple[list[str], np.ndarray]] = {}

    def _bank(self, name: str, prompts: dict[str, list[str]]):
        if name not in self._banks:
            self._banks[name] = self.embedder.prompt_bank(prompts)
        return self._banks[name]

    def _pick(self, emb, name, prompts):
        labs, bank = self._bank(name, prompts)
        p = zero_shot(emb, bank)[0]
        return labs[int(p.argmax())], p

    def _mask(self, img, photo_type: str) -> np.ndarray:
        seg = self.worn_seg if photo_type == "worn" else self.flat_seg
        if seg is None:
            seg = WornSegmenter() if photo_type == "worn" else FlatSegmenter()
            if photo_type == "worn":
                self.worn_seg = seg
            else:
                self.flat_seg = seg
        try:
            m = seg(img)
            if m.mean() > 0.002:
                return m
        except Exception:  # pragma: no cover - model/runtime problems fall back to plain background
            pass
        return self.fallback_seg(img)

    def analyse(self, img, size: str = "M", marker: tuple[np.ndarray, np.ndarray] | None = None) -> Guess:
        emb = self.embedder.images([img])
        photo_type, _ = self._pick(emb, "photo", L.PHOTO_TYPE_PROMPTS)

        labs, bank = self._bank("category", L.category_prompts())
        zs = zero_shot(emb, bank)
        hp = self.head.proba(emb) if self.head is not None else None
        p = ensemble(zs, labs, self.head, hp)[0]
        order = np.argsort(-p)
        category = labs[int(order[0])]
        top3 = [(labs[int(i)], float(p[int(i)])) for i in order[:3]]

        fabric, _ = self._pick(emb, "fabric", L.FABRIC_PROMPTS)
        embellish, _ = self._pick(emb, "embellish", L.EMBELLISH_PROMPTS)
        cat_cfg = config.categories()[category]
        if cat_cfg.get("embellish") == "heavy" and embellish == "none":
            embellish = "heavy"  # a bridal lehenga / sherwani is assumed heavy unless clearly plain

        mask = self._mask(img, photo_type)
        sleeve = None
        if marker is not None and photo_type == "flat":
            m = measure_with_marker(mask, marker[0], marker[1], cat_cfg["template"])
            flat_w, flat_l, source = m.flat_w, m.flat_l, "marker"
            sleeve = m.sleeve_len
        else:
            flat_w, flat_l, source = size_from_chart(category, size, mask, photo_type)
        conf = float(top3[0][1]) * SIZE_CONFIDENCE[source]
        return Guess(category, top3, photo_type, fabric, embellish, flat_w, flat_l, source, round(conf, 3), sleeve, mask)
