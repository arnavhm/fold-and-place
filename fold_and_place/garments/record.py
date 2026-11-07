"""The garment record: the single object every other module reads.

Vision writes it, the catalog writes it, the dashboard edits it. Folding and packing never look at
pixels - only at this record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from fold_and_place import config

SIZE_SOURCES = ("marker", "size_chart", "fabric_standard", "manual")


@dataclass
class Garment:
    id: str
    category: str
    label: str
    template: str
    flat_w: float  # cm, flat-lay width (body width for tops)
    flat_l: float  # cm, flat-lay length
    fabric: str
    base_layers: int = 2
    sleeve_len: float = 0.0
    sleeve_w: float = 0.0
    hem_w: float | None = None
    rise: float = 0.0
    size: str = "M"
    embellish: str = "none"  # none | light | heavy
    can_hang: bool = True
    hang_only: bool = False
    rod_in: float = 1.0
    hang_drop: float | None = None
    owner: str = ""
    value: float = 5.0  # importance 1-10 to its owner (private in the mechanism layer)
    wear_freq: float = 1.0  # wears per month
    size_source: str = "size_chart"
    confidence: float = 1.0
    image: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # ----- derived properties -----------------------------------------------------------
    @property
    def thickness_cm(self) -> float:
        return config.fabrics()[self.fabric]["thickness_mm"] / 10.0

    @property
    def crease_risk(self) -> float:
        return float(config.fabrics()[self.fabric]["crease_risk"])

    @property
    def fold_flat_only(self) -> bool:
        """Heavy embellished pieces are folded flat with tissue, never hung, rolled or filed."""
        return self.embellish == "heavy"

    @property
    def no_load_on_top(self) -> bool:
        """Nothing heavy may be stacked on embroidery (it crushes)."""
        return self.embellish == "heavy"

    @property
    def tissue_layers(self) -> bool:
        return self.embellish == "heavy" or self.fabric == "silk"

    @property
    def drop_cm(self) -> float:
        """Vertical space needed below the rod when hung."""
        return float(self.hang_drop if self.hang_drop is not None else self.flat_l)

    @property
    def is_drape(self) -> bool:
        return bool(self.extra.get("drape", False))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            thickness_cm=self.thickness_cm,
            crease_risk=self.crease_risk,
            fold_flat_only=self.fold_flat_only,
            no_load_on_top=self.no_load_on_top,
        )
        return d


def make_garment(
    category: str,
    gid: str,
    size: str = "M",
    *,
    owner: str = "",
    value: float = 5.0,
    wear_freq: float = 1.0,
    fabric: str | None = None,
    embellish: str | None = None,
    flat_w: float | None = None,
    flat_l: float | None = None,
    size_source: str | None = None,
    confidence: float = 1.0,
    image: str | None = None,
) -> Garment:
    """Build a garment from its category defaults, scaled to ``size``.

    Drapes (saree, dupatta, dhoti, lungi) keep their standard fabric size whatever ``size`` is.
    Explicit ``flat_w`` / ``flat_l`` (e.g. measured with a marker) override the defaults.
    """
    cats = config.categories()
    if category not in cats:
        raise KeyError(f"unknown category {category!r}")
    c = cats[category]
    drape = bool(c.get("drape", False))
    sw, sl = (1.0, 1.0) if drape else config.size_scale()[size]

    w = float(flat_w) if flat_w is not None else c["flat_w"] * sw
    length = float(flat_l) if flat_l is not None else c["flat_l"] * sl
    hem = c.get("hem_w")
    src = size_source or ("fabric_standard" if drape else "size_chart")
    if src not in SIZE_SOURCES:
        raise ValueError(f"size_source must be one of {SIZE_SOURCES}")

    return Garment(
        id=gid,
        category=category,
        label=c["label"],
        template=c["template"],
        flat_w=round(w, 2),
        flat_l=round(length, 2),
        fabric=fabric or c["fabric"],
        base_layers=int(c.get("base_layers", 2)),
        sleeve_len=float(c.get("sleeve_len", 0)) * (1.0 if drape else sw),
        sleeve_w=float(c.get("sleeve_w", 0)),
        hem_w=None if hem is None else float(hem) * (1.0 if drape else sw),
        rise=float(c.get("rise", 0)),
        size=size,
        embellish=embellish or c.get("embellish", "none"),
        can_hang=bool(c.get("can_hang", True)),
        hang_only=bool(c.get("hang_only", False)),
        rod_in=float(c.get("rod_in", 1.0)),
        hang_drop=c.get("hang_drop"),
        owner=owner,
        value=float(value),
        wear_freq=float(wear_freq),
        size_source=src,
        confidence=float(confidence),
        image=image,
        extra={"drape": drape},
    )
