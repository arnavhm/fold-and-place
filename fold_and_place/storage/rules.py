"""Household rules that limit where clothes may go, as an ``allowed`` map for the optimizer.

* personal shelves - a person's clothes go only in their own compartments or shared ones;
* pack winter clothes away - off-season clothes go only to hard-to-reach space (loft or
  anything outside everyone's easy-reach band), freeing the easy shelves;
* heavy things low - denim, heavy wool, suiting and heavily embroidered pieces never go in the
  loft (lifting them down from overhead is awkward and unsafe).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fold_and_place.garments.record import Garment
from fold_and_place.storage.wardrobe import Wardrobe

WINTER_CATEGORIES = {"hoodie", "sweater", "jacket", "nehru_jacket"}
WINTER_FABRICS = {"wool_knit_heavy", "fleece"}
HEAVY_FABRICS = {"denim", "wool_knit_heavy", "suiting", "brocade_zari"}


def is_winter(g: Garment) -> bool:
    return g.category in WINTER_CATEGORIES or g.fabric in WINTER_FABRICS or bool(g.extra.get("winter"))


def is_heavy(g: Garment) -> bool:
    return g.fabric in HEAVY_FABRICS or g.embellish == "heavy"


@dataclass
class Rules:
    personal: dict[str, set[str]] = field(default_factory=dict)  # person -> their compartment ids
    pack_winter: bool = False
    heavy_low: bool = False

    def active(self) -> bool:
        return bool(self.pack_winter or self.heavy_low or any(self.personal.values()))

    def allowed(self, gs: list[Garment], w: Wardrobe) -> dict[str, set[str]] | None:
        if not self.active():
            return None
        everyone = {c.id for c in w.compartments}
        owned = set().union(*self.personal.values()) if self.personal else set()
        shared = everyone - owned
        away = {c.id for c in w.compartments if c.kind == "loft" or c.zone_cost() > 1.0}
        out: dict[str, set[str]] = {}
        for g in gs:
            ok = set(everyone)
            if any(self.personal.values()):
                ok = shared | self.personal.get(g.owner, set())
            if self.pack_winter and is_winter(g):
                ok &= away
            if self.heavy_low and is_heavy(g):
                ok -= {c.id for c in w.compartments if c.kind == "loft"}
            out[g.id] = ok
        return out
