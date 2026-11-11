"""Generate every sensible fold option for a garment.

An option = a fold sequence + a storage mode:
  flat  folded packet lying flat, stacked on a shelf
  file  the same packet stood on edge in a row (KonMari "file fold"), for drawers
  roll  folded lengthwise, then rolled from the hem (military roll)
  hang  on a hanger on a rail (trousers and drapes folded once over the bar)

Options too big for any compartment are dropped; within each mode, options beaten on every
dimension by another (longer side, shorter side, height, creases) are dropped too.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from fold_and_place import config
from fold_and_place.folding.engine import (
    FoldState,
    FoldStep,
    fit_length,
    leg_over_leg,
    panels,
    sides_in,
)
from fold_and_place.garments.record import Garment

MAX_SIDE_CM = 60.0  # deepest compartment in any preset
MIN_SIDE_CM = 3.0
BOARD_CM = 23.0  # department-store board width (9 in), Make: magazine
BOARD_L_CM = 30.5  # department-store board length (12 in)


@dataclass
class FoldOption:
    id: str
    garment_id: str
    method: str
    mode: str  # flat | file | roll | hang
    steps: list[FoldStep] = field(default_factory=list)
    a: float = 0.0  # longer side of the folded packet (cm)
    b: float = 0.0  # shorter side (cm)
    h: float = 0.0  # stack height / thickness (cm)
    plies: int = 0  # max fabric plies anywhere in the packet
    creases: int = 0
    rod_in: float = 0.0  # hang only
    drop: float = 0.0  # hang only (cm)
    footprint_w: float = 0.0  # x extent after folding (animation frame)
    footprint_l: float = 0.0  # y extent after folding
    roll: bool = False
    tags: list[str] = field(default_factory=list)

    @property
    def volume(self) -> float:
        return self.a * self.b * self.h

    def summary(self) -> str:
        if self.mode == "hang":
            return f"{self.method} - hang ({self.rod_in:.2f} in of rod, {self.drop:.0f} cm drop)"
        return f"{self.method} - {self.mode}, {self.a:.0f} x {self.b:.0f} x {self.h:.1f} cm"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "garment_id": self.garment_id,
            "method": self.method,
            "mode": self.mode,
            "steps": [s.to_dict() for s in self.steps],
            "a": round(self.a, 2),
            "b": round(self.b, 2),
            "h": round(self.h, 2),
            "plies": self.plies,
            "creases": self.creases,
            "rod_in": self.rod_in,
            "drop": round(self.drop, 1),
            "roll": self.roll,
            "tags": self.tags,
        }


def packet(g: Garment, st: FoldState, k_loft: float | None = None) -> tuple[float, float, float, int]:
    """(long side, short side, height, plies) of a folded state."""
    k = config.k_loft() if k_loft is None else k_loft
    plies = st.max_plies()
    w, length = st.width, st.length
    return max(w, length), min(w, length), plies * g.thickness_cm * k, plies


def _flat_and_file(g: Garment, st: FoldState, method: str, tags: list[str]) -> list[FoldOption]:
    a, b, h, plies = packet(g, st)
    common = dict(
        garment_id=g.id,
        method=method,
        steps=list(st.steps),
        a=a,
        b=b,
        h=h,
        plies=plies,
        creases=len(st.steps),
        footprint_w=st.width,
        footprint_l=st.length,
    )
    out = [FoldOption(id="", mode="flat", tags=list(tags), **common)]
    if not g.fold_flat_only:
        out.append(FoldOption(id="", mode="file", tags=list(tags) + ["konmari"], **common))
    return out


def _roll(g: Garment, st: FoldState, method: str) -> FoldOption:
    """Roll the (width-folded) garment from the hem. Diameter from the area of the spiral."""
    k = config.k_loft()
    plies = st.max_plies()
    length, width = st.length, st.width
    t = plies * g.thickness_cm * k
    d = math.sqrt(4 * length * t / math.pi)
    steps = list(st.steps)
    return FoldOption(
        id="",
        garment_id=g.id,
        method=method,
        mode="roll",
        steps=steps,
        a=max(width, d),
        b=min(width, d),
        h=d,
        plies=plies,
        creases=len(steps),
        footprint_w=width,
        footprint_l=d,
        roll=True,
        tags=["roll"],
    )


def _hang(g: Garment) -> FoldOption:
    over_bar = g.hang_drop is not None and g.hang_drop < g.flat_l
    steps = []
    if over_bar:
        steps = [FoldStep("y", g.flat_l / 2, "lt", "Fold in half over the hanger bar.")]
    return FoldOption(
        id="",
        garment_id=g.id,
        method="Hang" + (" (folded over the bar)" if over_bar else ""),
        mode="hang",
        steps=steps,
        creases=len(steps),
        rod_in=g.rod_in,
        drop=g.drop_cm,
        tags=["hang"],
    )


def _tops(g: Garment) -> list[FoldOption]:
    out: list[FoldOption] = []
    base = FoldState.from_garment(g)
    W = g.flat_w
    keeps = {"thirds": W / 3, "board": min(BOARD_CM, 0.6 * W), "half": W / 2}
    for kname, keep in keeps.items():
        st_w = sides_in(base, keep)
        for n in (2, 3, 4):
            st = panels(st_w, "y", n)
            word = {2: "halves", 3: "thirds", 4: "quarters"}[n]
            method = f"Sides in to {keep:.0f} cm, {word} along the length"
            tags = []
            if kname == "thirds" and n == 3:
                tags += ["konmari-shape", "named"]
            out += _flat_and_file(g, st, method, tags)
        if kname == "board":
            st = fit_length(st_w, BOARD_L_CM)
            out += _flat_and_file(
                g, st, f"Department-store board fold ({keep:.0f} x {BOARD_L_CM:.1f} cm)", ["department-store", "named"]
            )
        if kname == "half" and not g.fold_flat_only:
            out.append(_roll(g, st_w, f"Sides in to {keep:.0f} cm, rolled from the hem"))
    return out


def _bottoms(g: Garment) -> list[FoldOption]:
    out: list[FoldOption] = []
    base = leg_over_leg(FoldState.from_garment(g))
    for nx in (1, 2):
        st_w = panels(base, "x", nx, "Width")
        for n in (2, 3, 4):
            st = panels(st_w, "y", n)
            word = {2: "halves", 3: "thirds", 4: "quarters"}[n]
            extra = ", folded in half lengthwise" if nx == 2 else ""
            out += _flat_and_file(g, st, f"Leg over leg{extra}, {word} along the length", [])
        if not g.fold_flat_only:
            out.append(_roll(g, st_w, f"Leg over leg{extra if nx == 2 else ''}, rolled from the hem"))
    return out


def _skirt(g: Garment) -> list[FoldOption]:
    out: list[FoldOption] = []
    base = FoldState.from_garment(g)
    for nx in (1, 2, 3):
        st_w = panels(base, "x", nx, "Width")
        for n in (1, 2, 3, 4):
            st = panels(st_w, "y", n)
            out += _flat_and_file(g, st, f"Width in {nx}, length in {n}", ["tissue"] if g.tissue_layers else [])
    return out


def _rect(g: Garment) -> list[FoldOption]:
    out: list[FoldOption] = []
    base = FoldState.from_garment(g)
    if g.is_drape:
        widths, lengths = (2, 3, 4), (4, 6, 8, 10, 12, 16, 20)
    else:
        widths, lengths = (1, 2, 3), (1, 2, 3, 4)
    for nx in widths:
        st_w = panels(base, "x", nx, "Lengthwise")
        for n in lengths:
            st = panels(st_w, "y", n, "Accordion" if g.is_drape else "")
            label = f"Lengthwise in {nx}, then {n} panels" if g.is_drape else f"Width in {nx}, length in {n}"
            out += _flat_and_file(g, st, label, ["drape"] if g.is_drape else [])
        if not g.fold_flat_only and (not g.is_drape or nx >= 3):
            out.append(_roll(g, st_w, f"Width in {nx}, rolled"))
    return out


GENERATORS = {"tee": _tops, "dress": _tops, "trousers": _bottoms, "skirt": _skirt, "rect": _rect}


def _fits_somewhere(o: FoldOption) -> bool:
    if o.mode == "hang":
        return True
    return MIN_SIDE_CM <= o.b and o.a <= MAX_SIDE_CM and o.h > 0


def _pareto(opts: list[FoldOption]) -> list[FoldOption]:
    """Drop options beaten on every dimension; named standard methods are always kept."""
    keep: list[FoldOption] = []
    for o in opts:
        if "named" in o.tags:
            keep.append(o)
            continue
        dominated = False
        for p in opts:
            if p is o or p.mode != o.mode:
                continue
            le = p.a <= o.a + 1e-6 and p.b <= o.b + 1e-6 and p.h <= o.h + 1e-6 and p.creases <= o.creases
            lt = p.a < o.a - 1e-6 or p.b < o.b - 1e-6 or p.h < o.h - 1e-6 or p.creases < o.creases
            if le and lt:
                dominated = True
                break
        if not dominated:
            keep.append(o)
    # drop exact duplicates (same mode and dimensions)
    seen, uniq = set(), []
    for o in keep:
        key = (o.mode, round(o.a, 1), round(o.b, 1), round(o.h, 2), o.creases)
        if key not in seen:
            seen.add(key)
            uniq.append(o)
    return uniq


def _signature(g: Garment) -> tuple:
    """Everything that changes the geometry or care of folding (not owner, value or id)."""
    return (
        g.template, round(g.flat_w, 2), round(g.flat_l, 2), g.fabric, g.base_layers,
        round(g.sleeve_len, 2), round(g.sleeve_w, 2), g.hem_w, g.rise, g.embellish,
        g.can_hang, g.hang_only, g.rod_in, g.hang_drop, g.is_drape, config.k_loft(),
    )


_CACHE: dict[tuple, list[FoldOption]] = {}


def generate_options(g: Garment, prune: bool = True) -> list[FoldOption]:
    """All fold options for ``g`` (pruned to the useful ones unless ``prune=False``).

    Identical garments (same signature) reuse the computed geometry; ids are per garment.
    """
    key = (_signature(g), prune)
    if key not in _CACHE:
        opts: list[FoldOption] = []
        if not g.hang_only:
            opts = GENERATORS[g.template](g)
        if g.can_hang and not g.fold_flat_only:
            opts.append(_hang(g))
        if prune:
            opts = _pareto([o for o in opts if _fits_somewhere(o)])
        _CACHE[key] = opts
    out = []
    for i, o in enumerate(_CACHE[key]):
        c = copy.copy(o)
        c.id, c.garment_id, c.tags = f"{g.id}#{i}", g.id, list(o.tags)
        out.append(c)
    return out
