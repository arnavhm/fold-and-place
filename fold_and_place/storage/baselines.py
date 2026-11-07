"""Baselines the optimizer is compared against (experiment E4).

B1 traditional  : everything folded flat (board fold / leg over leg / drapes in panels), stacked by
                  category on shelves and in drawers; only must-hang items hung.
B2 KonMari      : file-fold into drawer rows where it fits, hang dresses/jackets, rest flat.
B3 hang-first   : hang everything that can hang, fold the rest flat.
B4 greedy       : our full option set, garments in order of value per volume, cheapest spot first.
"""

from __future__ import annotations

from fold_and_place.folding.options import FoldOption, generate_options
from fold_and_place.garments.record import Garment
from fold_and_place.storage.greedy import Packer
from fold_and_place.storage.plan import Plan, fits, reach_cost
from fold_and_place.storage.wardrobe import Wardrobe

KONMARI_HANG = {"dress", "skirt", "jacket", "blazer", "gown", "kurti"}


def _opts(gs: list[Garment]) -> dict[str, list[FoldOption]]:
    return {g.id: generate_options(g) for g in gs}


def _compartments_top_down(w: Wardrobe, kinds: tuple[str, ...]):
    return sorted([c for c in w.compartments if c.kind in kinds], key=lambda c: (-c.z, c.x))


def _traditional_option(opts: list[FoldOption]) -> FoldOption | None:
    flat = [o for o in opts if o.mode == "flat"]
    if not flat:
        return None
    for key in ("Department-store", "Leg over leg, thirds", "then 16 panels", "then 12 panels"):
        hit = [o for o in flat if key in o.method]
        if hit:
            return hit[0]
    return min(flat, key=lambda o: (o.a, o.h))


def _place_flat(p: Packer, g: Garment, o: FoldOption, comps) -> bool:
    for c in comps:
        lay = "stack"
        if p.try_place(g, o, c, lay):
            return True
    return False


def b1_traditional(gs: list[Garment], w: Wardrobe) -> Plan:
    opts = _opts(gs)
    p = Packer(w, {g.id: g for g in gs})
    shelves = _compartments_top_down(w, ("shelf",)) + _compartments_top_down(w, ("drawer",)) + _compartments_top_down(w, ("loft",))
    rails = _compartments_top_down(w, ("rail_long", "rail_short"))
    for g in sorted(gs, key=lambda g: (g.category, g.owner, g.id)):
        if g.hang_only:
            hang = [o for o in opts[g.id] if o.mode == "hang"]
            for c in rails:
                if hang and p.try_place(g, hang[0], c):
                    break
            continue
        o = _traditional_option(opts[g.id])
        if o is not None:
            _place_flat(p, g, o, shelves)
    return p.plan("B1 traditional flat stacks")


def b2_konmari(gs: list[Garment], w: Wardrobe) -> Plan:
    opts = _opts(gs)
    p = Packer(w, {g.id: g for g in gs})
    drawers = _compartments_top_down(w, ("drawer",))
    shelves = _compartments_top_down(w, ("shelf",)) + _compartments_top_down(w, ("loft",))
    rails = _compartments_top_down(w, ("rail_long", "rail_short"))
    for g in sorted(gs, key=lambda g: (g.category, g.owner, g.id)):
        os_ = opts[g.id]
        hang = [o for o in os_ if o.mode == "hang"]
        if hang and (g.hang_only or g.category in KONMARI_HANG):
            if any(p.try_place(g, hang[0], c) for c in rails):
                continue
        files = sorted([o for o in os_ if o.mode == "file"], key=lambda o: ("konmari-shape" not in o.tags, o.h))
        if any(p.try_place(g, o, c) for o in files for c in drawers):
            continue
        flats = sorted([o for o in os_ if o.mode == "flat"], key=lambda o: ("konmari-shape" not in o.tags, o.a))
        placed = any(_place_flat(p, g, o, shelves) for o in flats)
        if not placed and hang:
            any(p.try_place(g, hang[0], c) for c in rails)
    return p.plan("B2 KonMari file fold")


def b3_hang_first(gs: list[Garment], w: Wardrobe) -> Plan:
    opts = _opts(gs)
    p = Packer(w, {g.id: g for g in gs})
    shelves = _compartments_top_down(w, ("shelf",)) + _compartments_top_down(w, ("drawer",)) + _compartments_top_down(w, ("loft",))
    rails = _compartments_top_down(w, ("rail_long", "rail_short"))
    for g in sorted(gs, key=lambda g: (g.category, g.owner, g.id)):
        hang = [o for o in opts[g.id] if o.mode == "hang"]
        if hang and any(p.try_place(g, hang[0], c) for c in rails):
            continue
        o = _traditional_option(opts[g.id])
        if o is not None:
            _place_flat(p, g, o, shelves)
    return p.plan("B3 hang everything possible")


def b4_greedy(
    gs: list[Garment],
    w: Wardrobe,
    values: dict[str, float] | None = None,
    lam_reach: float = 1.0,
    lam_crease: float = 1.0,
) -> Plan:
    """Value-density greedy over the full option set; each garment takes its cheapest feasible spot."""
    opts = _opts(gs)
    val = {g.id: (values or {}).get(g.id, g.value) for g in gs}
    p = Packer(w, {g.id: g for g in gs})

    def density(g: Garment) -> float:
        vols = [o.volume for o in opts[g.id] if o.mode != "hang"] or [1.0]
        return val[g.id] / max(min(vols), 1.0)

    for g in sorted(gs, key=lambda g: (-val[g.id], -density(g), g.id)):
        cands = []
        for o in opts[g.id]:
            for c in w.compartments:
                if fits(o, c):
                    cost = lam_reach * g.wear_freq * reach_cost(g, o, c) + lam_crease * g.crease_risk * o.creases
                    cands.append((cost, o.volume, o.id, o, c))
        cands.sort(key=lambda t: t[:3])
        for _, _, _, o, c in cands:
            if p.try_place(g, o, c):
                break
    return p.plan("B4 greedy")


BASELINES = {"B1": b1_traditional, "B2": b2_konmari, "B3": b3_hang_first, "B4": b4_greedy}
