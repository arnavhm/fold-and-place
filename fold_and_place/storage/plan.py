"""The plan: where every garment goes, in which fold. Shared by the optimizer, baselines,
checker, dashboard and mechanisms."""

from __future__ import annotations

from dataclasses import dataclass, field

from fold_and_place.folding.options import FoldOption
from fold_and_place.garments.record import Garment
from fold_and_place.storage.wardrobe import IN, MODE_FACTOR, Compartment, Wardrobe, golden_for

LAYOUT_MODES = {"stack": {"flat"}, "row": {"file", "roll"}, "rail": {"hang"}}
VALUE_WEIGHT = 20.0  # one point of garment value outweighs most retrieval / crease costs
OVERFLOW_REACH = 5.0  # effort per wear for a garment that did not fit (kept on a chair)


def stack_width(o: FoldOption, c: Compartment) -> float | None:
    """Width a flat packet takes along a shelf: short side across if the long side fits the depth."""
    if o.a <= c.d + 1e-9:
        return o.b
    if o.b <= c.d + 1e-9:
        return o.a
    return None


def fits(o: FoldOption, c: Compartment, layout: str | None = None) -> bool:
    """Can option ``o`` go in compartment ``c`` at all (ignoring what else is there)?"""
    lay = layout or c.layout
    if o.mode not in LAYOUT_MODES[lay]:
        return False
    if lay == "stack":
        sw = stack_width(o, c)
        return sw is not None and sw <= c.w + 1e-9 and o.h <= c.h + 1e-9
    if lay == "row":
        return o.a <= c.d + 1e-9 and o.b <= c.h + 1e-9 and o.h <= c.w + 1e-9
    return o.rod_in * IN <= c.w + 1e-9 and o.drop <= c.h + 1e-9


def reach_cost(g: Garment, o: FoldOption, c: Compartment) -> float:
    """Effort to take this garment out once: zone cost (for its owner) x mode factor."""
    return c.zone_cost(golden_for(g.owner)) * MODE_FACTOR[o.mode]


@dataclass
class Lane:
    """One stack (shelf/loft/drawer), one row (drawer) or one rail, with items bottom->top or
    front->back or left->right."""

    compartment: str
    layout: str  # stack | row | rail
    items: list[str] = field(default_factory=list)  # garment ids


@dataclass
class Plan:
    wardrobe: Wardrobe
    garments: dict[str, Garment]
    choice: dict[str, FoldOption]  # garment id -> chosen option (stored garments only)
    lanes: list[Lane]
    overflow: list[str]
    method: str = ""
    status: str = ""
    solve_s: float = 0.0
    gap: float | None = None  # relative optimality gap reported by the solver (None = heuristic)

    # ---------- lookups ----------
    def lane_of(self, gid: str) -> Lane | None:
        for lane in self.lanes:
            if gid in lane.items:
                return lane
        return None

    def compartment_of(self, gid: str) -> Compartment | None:
        lane = self.lane_of(gid)
        return self.wardrobe.get(lane.compartment) if lane else None

    @property
    def stored(self) -> list[str]:
        return [gid for lane in self.lanes for gid in lane.items]

    # ---------- metrics ----------
    def metrics(self, lam_reach: float = 1.0, lam_crease: float = 1.0) -> dict:
        stored = self.stored
        gs = self.garments
        value = sum(gs[g].value for g in stored)
        effort = sum(gs[g].wear_freq * reach_cost(gs[g], self.choice[g], self.compartment_of(g)) for g in stored)
        crease = sum(gs[g].crease_risk * self.choice[g].creases for g in stored)
        packet_vol = sum(self.choice[g].volume for g in stored if self.choice[g].mode != "hang")
        rod = sum(self.choice[g].rod_in * IN for g in stored if self.choice[g].mode == "hang")
        overflow_effort = sum(gs[g].wear_freq * OVERFLOW_REACH for g in self.overflow)
        return {
            "method": self.method,
            "stored": len(stored),
            "overflow": len(self.overflow),
            "stored_value": value,
            "total_value": sum(g.value for g in gs.values()),
            "retrieval_effort": round(effort, 2),
            "effort_incl_overflow": round(effort + overflow_effort, 2),
            "crease_load": round(crease, 2),
            "packet_fill_pct": round(100 * packet_vol / self.wardrobe.packet_volume, 1),
            "rod_fill_pct": round(100 * rod / self.wardrobe.rod_cm, 1) if self.wardrobe.rod_cm else 0.0,
            # same objective the optimizer maximises (overflow items cost 5 per wear: a chair pile)
            "objective": round(
                VALUE_WEIGHT * value - lam_reach * (effort + overflow_effort) - lam_crease * crease, 2
            ),
            "solve_s": round(self.solve_s, 2),
            "gap": self.gap,
            "status": self.status,
        }

    def per_owner_value(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for gid in self.stored:
            g = self.garments[gid]
            out[g.owner] = out.get(g.owner, 0.0) + g.value
        return out

    def rows(self) -> list[dict]:
        """One row per garment for tables and CSVs."""
        out = []
        for gid, g in self.garments.items():
            o = self.choice.get(gid)
            lane = self.lane_of(gid)
            c = self.wardrobe.get(lane.compartment) if lane else None
            out.append(
                {
                    "garment": gid,
                    "owner": g.owner,
                    "category": g.label,
                    "value": g.value,
                    "wears_per_month": g.wear_freq,
                    "fold": o.method if o else "",
                    "mode": o.mode if o else "",
                    "folded_cm": (f"{o.a:.0f} x {o.b:.0f} x {o.h:.1f}" if o and o.mode != "hang" else (f"{o.rod_in:.2f} in rod" if o else "")),
                    "compartment": c.id if c else "OVERFLOW",
                    "position": (lane.items.index(gid) + 1) if lane else None,
                    "reason": explain(g, o, c, lane) if lane else "Did not fit: lower value than what was stored.",
                }
            )
        return out


def explain(g: Garment, o: FoldOption, c: Compartment, lane: Lane) -> str:
    zone = c.zone_cost(golden_for(g.owner))
    where = {1.0: "in the easy-reach zone", 1.5: "a short bend or stretch away", 3.0: "hard to reach (floor or loft)"}.get(zone, "")
    bits = [f"Worn {g.wear_freq:g}x/month, value {g.value:g}: {c.kind.replace('_', ' ')} {where}."]
    if o.mode == "file":
        bits.append("Filed on edge so it is visible and comes out without disturbing others.")
    elif o.mode == "roll":
        bits.append("Rolled: compact and fewer sharp creases.")
    elif o.mode == "hang":
        bits.append("Hung: no fold creases.")
    else:
        bits.append("Folded flat in a stack.")
    if g.no_load_on_top:
        bits.append("Heavy embroidery: kept on top of its stack with tissue, nothing above it.")
    return " ".join(bits)
