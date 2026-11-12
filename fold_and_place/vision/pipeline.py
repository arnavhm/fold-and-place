"""Photo -> garment record.

1. photo type (flat-lay / worn / folded), zero-shot
2. mask: rembg for flat-lay, SegFormer clothes parser for worn photos, plain-background fallback
3. category: zero-shot FashionSigLIP, averaged with the trained head when one exists
4. fabric and embellishment, zero-shot
5. size, with the source recorded:
   - a reference in the photo (an A4 sheet or ArUco marker, found automatically) -> measured;
   - a flat-lay photo without one -> the garment's own proportions, scaled to the size label's
     chart length ("approx");
   - worn photos -> size chart; drapes -> fabric standard.
   The garment is turned upright first, and its outline (cm) is kept so we fold *this* garment.
Every component is injectable, so tests run with fakes and no model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fold_and_place import config
from fold_and_place.garments.record import Garment, make_garment
from fold_and_place.vision import labels as L
from fold_and_place.vision.measure import drop_marker, homography, size_from_chart
from fold_and_place.vision.models import (
    Embedder,
    FlatSegmenter,
    LinearHead,
    PlainSegmenter,
    WornSegmenter,
    ensemble,
    zero_shot,
)
from fold_and_place.vision.reference import find_reference, main_part
from fold_and_place.vision.shape import silhouette

HEAD_PATH = config.ROOT / "models" / "head.npz"
SIZE_CONFIDENCE = {"marker": 0.9, "fabric_standard": 0.7, "size_chart": 0.6, "approx": 0.6}
SOURCE_TEXT = {
    "marker": "measured from the photo (reference sheet found)",
    "approx": "approximate: the photo's proportions, scaled to the size label",
    "size_chart": "from the size chart for the label size",
    "fabric_standard": "standard fabric size for this garment",
    "manual": "entered by hand",
}


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
    outline: list[tuple[float, float]] | None = None
    reference: tuple[np.ndarray, np.ndarray, str] | None = field(default=None, repr=False)
    upright: np.ndarray | None = field(default=None, repr=False)  # top-down mask, ``upright_res`` cm/cell
    upright_res: float = 0.5
    hem_w: float | None = None

    @property
    def source_text(self) -> str:
        return SOURCE_TEXT.get(self.size_source, self.size_source)

    @property
    def needs_check(self) -> bool:
        """Ask the person to confirm: unsure category, or a size that is only approximate."""
        return self.confidence < 0.45 or (len(self.top3) > 1 and self.top3[0][1] - self.top3[1][1] < 0.15)

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
            outline=self.outline,
        )
        if self.sleeve_len is not None and g.template in ("tee", "dress"):
            g.sleeve_len = self.sleeve_len
        if self.hem_w and g.template in ("dress", "skirt"):
            g.hem_w = round(float(self.hem_w), 1)
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

    def analyse(self, img, size: str = "M", marker: tuple[np.ndarray, np.ndarray] | None = None,
                auto_reference: bool = True, category: str | None = None) -> Guess:
        """``marker``: known reference corners (px, cm); otherwise one is looked for in the photo.
        ``category``: skip classification (the person corrected it)."""
        emb = self.embedder.images([img])
        photo_type, _ = self._pick(emb, "photo", L.PHOTO_TYPE_PROMPTS)
        ref = None
        if marker is not None:
            ref = (np.asarray(marker[0], float), np.asarray(marker[1], float), "given")
        elif auto_reference:
            ref = find_reference(img)
        if ref is not None:
            photo_type = "flat"  # a sheet of paper on the floor means a flat-lay photo

        labs, bank = self._bank("category", L.category_prompts())
        zs = zero_shot(emb, bank)
        hp = self.head.proba(emb) if self.head is not None else None
        p = ensemble(zs, labs, self.head, hp)[0]
        order = np.argsort(-p)
        top3 = [(labs[int(i)], float(p[int(i)])) for i in order[:3]]
        if category is None:
            category = labs[int(order[0])]
        else:
            top3 = [(category, 1.0)] + [t for t in top3 if t[0] != category][:2]

        fabric, _ = self._pick(emb, "fabric", L.FABRIC_PROMPTS)
        embellish, _ = self._pick(emb, "embellish", L.EMBELLISH_PROMPTS)
        cat_cfg = config.categories()[category]
        if cat_cfg.get("embellish") == "heavy" and embellish == "none":
            embellish = "heavy"  # a bridal lehenga / sherwani is assumed heavy unless clearly plain

        mask = self._mask(img, photo_type)
        if ref is not None:
            side = np.linalg.norm(ref[0][1] - ref[0][0])
            mask = drop_marker(mask, ref[0], grow_px=max(6.0, 0.05 * side))
        mask = main_part(mask)
        sleeve, outline, upright, res, hem = None, None, None, 0.5, None
        chart = make_garment(category, "probe", size)
        if photo_type == "flat" and mask.any():
            H = homography(ref[0], ref[1]) if ref is not None else None
            try:
                sil = silhouette(mask, cat_cfg["template"], H, length_cm=chart.flat_l)
                flat_w, flat_l = round(float(sil.measured.flat_w), 1), round(float(sil.measured.flat_l), 1)
                sleeve = round(float(sil.measured.sleeve_len), 1) if sil.measured.sleeve_len else None
                hem = sil.measured.hem_w
                outline, upright, res = sil.outline, sil.grid, sil.res
                source = "marker" if ref is not None else ("fabric_standard" if chart.is_drape else "approx")
            except ValueError:  # an impossible size: wrong reference, fall back
                flat_w, flat_l, source = size_from_chart(category, size, mask, photo_type)
                ref = None
        else:
            flat_w, flat_l, source = size_from_chart(category, size, mask, photo_type)
        conf = float(top3[0][1]) * SIZE_CONFIDENCE[source]
        return Guess(category, top3, photo_type, fabric, embellish, flat_w, flat_l, source, round(conf, 3), sleeve,
                     mask, outline, ref, upright, res, hem)


def measure_photo(img, category: str, size: str = "M", segmenter=None) -> Guess:
    """Measuring only (no recognition models): the person says what the garment is; the photo
    gives the size and outline. Uses rembg when installed, else the plain-background segmenter."""
    cat_cfg = config.categories()[category]
    ref = find_reference(img)
    if segmenter is None:
        try:
            segmenter = FlatSegmenter()
            mask = segmenter(img)
            if mask.mean() < 0.002:
                raise ValueError
        except Exception:  # rembg missing or failed
            mask = PlainSegmenter()(img)
    else:
        mask = segmenter(img)
    if ref is not None:
        side = np.linalg.norm(ref[0][1] - ref[0][0])
        mask = drop_marker(mask, ref[0], grow_px=max(6.0, 0.05 * side))
    mask = main_part(mask)
    chart = make_garment(category, "probe", size)
    H = homography(ref[0], ref[1]) if ref is not None else None
    try:
        sil = silhouette(mask, cat_cfg["template"], H, length_cm=chart.flat_l)
    except ValueError:
        w, length, src = size_from_chart(category, size, mask, "flat")
        return Guess(category, [(category, 1.0)], "flat", chart.fabric, chart.embellish, w, length, src, 0.6, None, mask)
    m = sil.measured
    source = "marker" if ref is not None else ("fabric_standard" if chart.is_drape else "approx")
    return Guess(category, [(category, 1.0)], "flat", chart.fabric, chart.embellish, round(float(m.flat_w), 1),
                 round(float(m.flat_l), 1), source, SIZE_CONFIDENCE[source],
                 round(float(m.sleeve_len), 1) if m.sleeve_len else None, mask, sil.outline, ref, sil.grid, sil.res, m.hem_w)
