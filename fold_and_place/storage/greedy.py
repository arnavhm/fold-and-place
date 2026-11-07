"""First-fit packer used by the baselines and as the optimizer's warm start / fallback."""

from __future__ import annotations

from dataclasses import dataclass, field

from fold_and_place.folding.options import FoldOption
from fold_and_place.garments.record import Garment
from fold_and_place.storage.plan import IN, Lane, Plan, fits, stack_width
from fold_and_place.storage.wardrobe import Compartment, Wardrobe


@dataclass
class _LaneState:
    lane: Lane
    span: float = 0.0  # stack: lane width; row: lane depth
    fill: float = 0.0  # stack: height used; row: thickness used; rail: rod used
    heavy: bool = False


@dataclass
class Packer:
    wardrobe: Wardrobe
    garments: dict[str, Garment]
    lanes: dict[str, list[_LaneState]] = field(default_factory=dict)
    layout: dict[str, str] = field(default_factory=dict)
    choice: dict[str, FoldOption] = field(default_factory=dict)

    def _span_used(self, cid: str) -> float:
        return sum(s.span for s in self.lanes.get(cid, []))

    def try_place(self, g: Garment, o: FoldOption, c: Compartment, layout: str | None = None) -> bool:
        lay = layout or c.layout
        if self.layout.get(c.id, lay) != lay or not fits(o, c, lay):
            return False
        if c.kind == "loft" and g.fold_flat_only:
            return False
        states = self.lanes.setdefault(c.id, [])
        if lay == "rail":
            if not states:
                states.append(_LaneState(Lane(c.id, "rail")))
            st = states[0]
            need = o.rod_in * IN
            if st.fill + need > c.w + 1e-9:
                return False
            st.fill += need
            st.lane.items.append(g.id)
        else:
            span_item = stack_width(o, c) if lay == "stack" else o.a
            cap_span = c.w if lay == "stack" else c.d
            cap_fill = c.h if lay == "stack" else c.w
            used = self._span_used(c.id)
            best, best_inc = None, None
            for st in states:
                if st.fill + o.h > cap_fill + 1e-9 or (g.no_load_on_top and st.heavy):
                    continue
                inc = max(0.0, span_item - st.span)
                if used + inc > cap_span + 1e-9:
                    continue
                if best is None or inc < best_inc or (inc == best_inc and st.fill > best.fill):
                    best, best_inc = st, inc
            if best is None:
                if used + span_item > cap_span + 1e-9 or o.h > cap_fill + 1e-9:
                    return False
                best = _LaneState(Lane(c.id, lay))
                states.append(best)
            best.span = max(best.span, span_item)
            best.fill += o.h
            best.heavy = best.heavy or g.no_load_on_top
            best.lane.items.append(g.id)
        self.layout[c.id] = lay
        self.choice[g.id] = o
        return True

    def plan(self, method: str) -> Plan:
        lanes = [s.lane for sts in self.lanes.values() for s in sts if s.lane.items]
        stored = {gid for lane in lanes for gid in lane.items}
        plan = Plan(
            self.wardrobe,
            self.garments,
            {k: v for k, v in self.choice.items() if k in stored},
            lanes,
            [gid for gid in self.garments if gid not in stored],
            method=method,
            status="heuristic",
        )
        order_lanes(plan)
        return plan


def order_lanes(plan: Plan) -> None:
    """Stacks: heavy embroidered piece on top, otherwise most-worn on top.
    Rows: most-worn at the front. Rails: most-worn first."""
    gs = plan.garments
    for lane in plan.lanes:
        if lane.layout == "stack":
            lane.items.sort(key=lambda g: (gs[g].no_load_on_top, gs[g].wear_freq, gs[g].value))
        else:
            lane.items.sort(key=lambda g: (-gs[g].wear_freq, -gs[g].value))
