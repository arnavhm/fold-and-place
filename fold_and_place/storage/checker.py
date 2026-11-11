"""Independent plan checker: re-verifies every placement from scratch.

Nothing reaches the dashboard or the experiments without passing this, so a bug in the optimizer
or a baseline cannot hide as an impossible plan.
"""

from __future__ import annotations

from fold_and_place.storage.plan import IN, LAYOUT_MODES, Plan, fits, stack_width

TOL = 1e-6


def check(plan: Plan) -> list[str]:
    """Return a list of violations (empty = valid)."""
    errs: list[str] = []
    seen: set[str] = set()
    by_comp: dict[str, list] = {}
    for lane in plan.lanes:
        by_comp.setdefault(lane.compartment, []).append(lane)
        for gid in lane.items:
            if gid in seen:
                errs.append(f"{gid} placed twice")
            seen.add(gid)
            if gid not in plan.garments:
                errs.append(f"{gid} unknown garment")
            o = plan.choice.get(gid)
            if o is None or o.garment_id != gid:
                errs.append(f"{gid} has no matching fold option")
    for gid in plan.overflow:
        if gid in seen:
            errs.append(f"{gid} both stored and overflow")
    missing = set(plan.garments) - seen - set(plan.overflow)
    if missing:
        errs.append(f"garments neither stored nor overflow: {sorted(missing)[:5]}")
    if errs:
        return errs

    for cid, lanes in by_comp.items():
        c = plan.wardrobe.get(cid)
        layouts = {lane.layout for lane in lanes}
        if len(layouts) > 1:
            errs.append(f"{cid}: mixes {layouts}")
            continue
        lay = layouts.pop()
        for lane in lanes:
            for gid in lane.items:
                o = plan.choice[gid]
                g = plan.garments[gid]
                if o.mode not in LAYOUT_MODES[lay]:
                    errs.append(f"{gid}: mode {o.mode} not allowed in a {lay}")
                elif not fits(o, c, lay):
                    errs.append(f"{gid}: option does not fit {cid}")
                if c.kind == "loft" and g.fold_flat_only:
                    errs.append(f"{gid}: heavy piece in the loft")
        if lay == "stack":
            width = 0.0
            for lane in lanes:
                ws = [stack_width(plan.choice[g], c) or 1e9 for g in lane.items]
                width += max(ws, default=0.0)
                hsum = sum(plan.choice[g].h for g in lane.items)
                if hsum > c.h + TOL:
                    errs.append(f"{cid}: stack height {hsum:.1f} > {c.h}")
                heavy = [g for g in lane.items if plan.garments[g].no_load_on_top]
                if len(heavy) > 1:
                    errs.append(f"{cid}: two heavy pieces in one stack")
                if heavy and lane.items[-1] != heavy[0]:
                    errs.append(f"{cid}: heavy piece {heavy[0]} not on top of its stack")
            if width > c.w + TOL:
                errs.append(f"{cid}: stacks need {width:.1f} cm > width {c.w}")
        elif lay == "row":
            depth = 0.0
            for lane in lanes:
                depth += max((plan.choice[g].a for g in lane.items), default=0.0)
                tsum = sum(plan.choice[g].h for g in lane.items)
                if tsum > c.w + TOL:
                    errs.append(f"{cid}: row thickness {tsum:.1f} > width {c.w}")
            if depth > c.d + TOL:
                errs.append(f"{cid}: rows need {depth:.1f} cm > depth {c.d}")
        else:
            rod = sum(plan.choice[g].rod_in * IN for lane in lanes for g in lane.items)
            if rod > c.w + TOL:
                errs.append(f"{cid}: rod {rod:.1f} cm > {c.w}")
    return errs


def assert_valid(plan: Plan) -> Plan:
    errs = check(plan)
    if errs:
        raise ValueError(f"invalid plan ({plan.method}): " + "; ".join(errs[:8]))
    return plan
