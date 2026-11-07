"""Exact wardrobe packing with OR-Tools CP-SAT.

Shelves and drawers are the same problem turned 90 degrees:
  shelf / loft : packets in side-by-side STACKS.  stack width = widest packet; widths sum <= shelf
                 width; heights in a stack sum <= clearance.
  drawer       : packets filed on edge in front-to-back ROWS. row depth = deepest packet; depths
                 sum <= drawer depth; thicknesses in a row sum <= drawer width.
  rail         : hanger rod space sums <= rod length; each garment's drop <= rail drop.

Decision: for every garment, at most one (fold option, lane). Objective (maximised):
  sum  20 * value  +  lam_reach * wears * (5 - reach)  -  lam_crease * crease_risk * creases
i.e. store as much value as possible, then make frequently worn things easy to reach (5 = effort of
an item left on a chair), then avoid creasing delicate fabrics. Sizes are integer millimetres:
items rounded UP, capacities rounded DOWN, so every solver answer passes the checker.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from fold_and_place.folding.options import FoldOption, generate_options
from fold_and_place.garments.record import Garment
from fold_and_place.storage.baselines import BASELINES, b4_greedy
from fold_and_place.storage.greedy import order_lanes
from fold_and_place.storage.plan import (
    IN,
    OVERFLOW_REACH,
    VALUE_WEIGHT,
    Lane,
    Plan,
    fits,
    reach_cost,
    stack_width,
)
from fold_and_place.storage.wardrobe import Wardrobe

MAX_LANES = 4
OPTIONS_PER_MODE = 4
SCALE = 100  # objective coefficients are scaled to integers


def _mm_up(cm: float) -> int:
    return int(math.ceil(cm * 10 - 1e-6))


def _mm_down(cm: float) -> int:
    return int(math.floor(cm * 10 + 1e-6))


def shortlist(opts: list[FoldOption], per_mode: int = OPTIONS_PER_MODE) -> list[FoldOption]:
    """Keep a few options per mode: the narrowest, the thinnest, the fewest creases (+ named)."""
    out: list[FoldOption] = []
    for mode in ("flat", "file", "roll", "hang"):
        group = [o for o in opts if o.mode == mode]
        if not group:
            continue
        picks = [
            min(group, key=lambda o: (o.b, o.h)),
            min(group, key=lambda o: (o.h, o.b)),
            min(group, key=lambda o: (o.creases, o.a)),
            min(group, key=lambda o: (o.a * o.b, o.h)),
        ]
        picks += [o for o in group if "named" in o.tags]
        seen = []
        for o in picks:
            if o not in seen:
                seen.append(o)
        out += seen[: max(per_mode, len([o for o in seen if "named" in o.tags]))]
    return out


@dataclass
class SolveConfig:
    time_limit_s: float = 30.0
    workers: int = 8
    lam_reach: float = 1.0
    lam_crease: float = 1.0
    seed: int = 0
    warm_start: bool = True


def solve(
    gs: list[Garment],
    w: Wardrobe,
    values: dict[str, float] | None = None,
    cfg: SolveConfig | None = None,
) -> Plan:
    """Optimal (or best found within the time limit) plan. ``values`` overrides garment values
    (used by the mechanisms, where values are what people REPORT)."""
    cfg = cfg or SolveConfig()
    t0 = time.time()
    garments = {g.id: g for g in gs}
    val = {g.id: float((values or {}).get(g.id, g.value)) for g in gs}
    if not gs:
        return Plan(w, garments, {}, [], [], method="CP-SAT optimizer", status="OPTIMAL", gap=0.0)
    options = {g.id: shortlist(generate_options(g)) for g in gs}

    # The model must contain every baseline's plan, so the optimum can never be worse than a
    # baseline: add the options and lane counts the baselines used, and start from the best one.
    seeds: list[Plan] = []
    if cfg.warm_start:
        seeds = [
            fn(gs, w) if name != "B4" else b4_greedy(gs, w, values=val, lam_reach=cfg.lam_reach, lam_crease=cfg.lam_crease)
            for name, fn in BASELINES.items()
        ]
        for sp in seeds:
            for gid, o in sp.choice.items():
                if not any(c.method == o.method and c.mode == o.mode for c in options[gid]):
                    options[gid].append(o)
    seed_lanes: dict[str, int] = {}
    for sp in seeds:
        counts: dict[str, int] = {}
        for lane in sp.lanes:
            counts[lane.compartment] = counts.get(lane.compartment, 0) + 1
        for cid, n in counts.items():
            seed_lanes[cid] = max(seed_lanes.get(cid, 0), n)

    # ---- lanes per compartment -------------------------------------------------------------
    lanes: list[tuple[str, str, int]] = []  # (compartment id, layout, index)
    for c in w.compartments:
        if c.layout == "rail":
            lanes.append((c.id, "rail", 0))
            continue
        spans = []
        for g in gs:
            for o in options[g.id]:
                if fits(o, c) and not (c.kind == "loft" and g.fold_flat_only):
                    spans.append(stack_width(o, c) if c.layout == "stack" else o.a)
        if not spans:
            continue
        cap = c.w if c.layout == "stack" else c.d
        k = max(1, min(MAX_LANES, int(cap // max(min(spans), 1e-6))), seed_lanes.get(c.id, 0))
        lanes += [(c.id, c.layout, i) for i in range(k)]

    m = cp_model.CpModel()
    x: dict[tuple[str, int, int], cp_model.IntVar] = {}
    coef: dict[tuple[str, int, int], int] = {}
    by_lane: dict[int, list[tuple[str, int, int]]] = {i: [] for i in range(len(lanes))}
    by_g: dict[str, list[tuple[str, int, int]]] = {g.id: [] for g in gs}
    for li, (cid, _lay, _) in enumerate(lanes):
        c = w.get(cid)
        for g in gs:
            if c.kind == "loft" and g.fold_flat_only:
                continue
            for oi, o in enumerate(options[g.id]):
                if not fits(o, c):
                    continue
                key = (g.id, oi, li)
                x[key] = m.new_bool_var(f"x[{g.id},{oi},{li}]")
                reach = reach_cost(g, o, c)
                score = (
                    VALUE_WEIGHT * val[g.id]
                    + cfg.lam_reach * g.wear_freq * (OVERFLOW_REACH - reach)
                    - cfg.lam_crease * g.crease_risk * o.creases
                )
                coef[key] = int(round(SCALE * score))
                by_lane[li].append(key)
                by_g[g.id].append(key)

    for keys in by_g.values():
        if keys:
            m.add_at_most_one(x[k] for k in keys)

    # ---- capacity constraints --------------------------------------------------------------
    comp_lanes: dict[str, list[int]] = {}
    for li, (cid, _, _) in enumerate(lanes):
        comp_lanes.setdefault(cid, []).append(li)
    for cid, lis in comp_lanes.items():
        c = w.get(cid)
        if c.layout == "rail":
            keys = [k for li in lis for k in by_lane[li]]
            m.add(sum(_mm_up(options[k[0]][k[1]].rod_in * IN) * x[k] for k in keys) <= _mm_down(c.w))
            continue
        cap_span = _mm_down(c.w if c.layout == "stack" else c.d)
        cap_fill = _mm_down(c.h if c.layout == "stack" else c.w)
        span_vars = []
        for li in lis:
            keys = by_lane[li]
            sv = m.new_int_var(0, cap_span, f"span[{li}]")
            span_vars.append(sv)
            for k in keys:
                o = options[k[0]][k[1]]
                span = stack_width(o, c) if c.layout == "stack" else o.a
                m.add(sv >= _mm_up(span) * x[k])
            m.add(sum(_mm_up(options[k[0]][k[1]].h) * x[k] for k in keys) <= cap_fill)
            heavy = [x[k] for k in keys if garments[k[0]].no_load_on_top]
            if len(heavy) > 1:
                m.add(sum(heavy) <= 1)
        m.add(sum(span_vars) <= cap_span)
        for a, b in zip(span_vars, span_vars[1:]):  # symmetry breaking: lanes widest first
            m.add(a >= b)

    m.maximize(sum(coef[k] * x[k] for k in x))

    # ---- warm start from the best baseline --------------------------------------------------
    if seeds:
        best = max(seeds, key=lambda sp: welfare(sp, val, cfg))
        hint = _hint_from_plan(best, lanes, options)
        for k, var in x.items():
            m.add_hint(var, 1 if k in hint else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cfg.time_limit_s
    solver.parameters.num_workers = cfg.workers
    solver.parameters.random_seed = cfg.seed
    status = solver.solve(m)
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        plan = b4_greedy(gs, w, values=val, lam_reach=cfg.lam_reach, lam_crease=cfg.lam_crease)
        plan.method, plan.status, plan.solve_s = "greedy fallback", f"CP-SAT {status_name}", time.time() - t0
        return plan

    choice: dict[str, FoldOption] = {}
    lane_objs: dict[int, Lane] = {}
    for k, var in x.items():
        if solver.value(var):
            gid, oi, li = k
            choice[gid] = options[gid][oi]
            cid, lay, _ = lanes[li]
            lane_objs.setdefault(li, Lane(cid, lay)).items.append(gid)
    obj, bound = solver.objective_value, solver.best_objective_bound
    gap = 0.0 if status == cp_model.OPTIMAL else abs(bound - obj) / max(1.0, abs(obj))
    plan = Plan(
        w,
        garments,
        choice,
        [lane_objs[i] for i in sorted(lane_objs)],
        [g.id for g in gs if g.id not in choice],
        method="CP-SAT optimizer",
        status=status_name,
        solve_s=time.time() - t0,
        gap=round(gap, 5),
    )
    plan.objective_scaled = obj  # type: ignore[attr-defined]
    order_lanes(plan)
    return plan


def _hint_from_plan(plan: Plan, lanes, options) -> set[tuple[str, int, int]]:
    """Map a greedy plan onto (garment, option index, lane index) keys, widest lane first."""
    hint = set()
    free: dict[str, list[int]] = {}
    for li, (cid, _, _) in enumerate(lanes):
        free.setdefault(cid, []).append(li)
    for lane in sorted(plan.lanes, key=lambda ln: (ln.compartment, -_lane_span(plan, ln))):
        if not free.get(lane.compartment):
            continue
        li = free[lane.compartment].pop(0)
        for gid in lane.items:
            o = plan.choice[gid]
            for oi, cand in enumerate(options[gid]):
                if cand.method == o.method and cand.mode == o.mode:
                    hint.add((gid, oi, li))
                    break
    return hint


def _lane_span(plan: Plan, lane: Lane) -> float:
    c = plan.wardrobe.get(lane.compartment)
    if lane.layout == "stack":
        return max((stack_width(plan.choice[g], c) or 0.0) for g in lane.items)
    if lane.layout == "row":
        return max(plan.choice[g].a for g in lane.items)
    return 0.0


def welfare(plan: Plan, values: dict[str, float] | None = None, cfg: SolveConfig | None = None) -> float:
    """The objective of ``plan`` (unscaled), with optional value overrides."""
    cfg = cfg or SolveConfig()
    total = 0.0
    for gid in plan.stored:
        g = plan.garments[gid]
        o = plan.choice[gid]
        v = (values or {}).get(gid, g.value)
        total += VALUE_WEIGHT * v + cfg.lam_reach * g.wear_freq * (
            OVERFLOW_REACH - reach_cost(g, o, plan.compartment_of(gid))
        ) - cfg.lam_crease * g.crease_risk * o.creases
    return total
