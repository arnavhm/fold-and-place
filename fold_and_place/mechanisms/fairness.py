"""Fairness measures for shared wardrobe space.

People do not want each other's clothes - they compete for SPACE. So envy is measured on space:

  K_i(space)  = the most value person i could store in that much space (packet volume + rod),
                each of i's garments taking its smallest packet or its hanger (a knapsack;
                geometry relaxed to volume, so it is optimistic - the same for everyone).
  envy_ij     = max(0, K_i(space j got) - K_i(space i got))
  EF1 holds   if, for every i, j, removing the single largest item from j's space removes envy.
  Proportional share PS_i = value i stores with 1/n of every compartment (exact solve).
"""

from __future__ import annotations

from functools import lru_cache

from ortools.sat.python import cp_model

from fold_and_place.folding import generate_options
from fold_and_place.mechanisms.mechanisms import EqualSplit, Outcome
from fold_and_place.mechanisms.scenario import Scenario
from fold_and_place.storage.plan import IN, Plan


def _items(sc: Scenario, agent: str, values: dict[str, float]):
    out = []
    for g in sc.owned(agent):
        opts = generate_options(g)
        packets = [o.volume for o in opts if o.mode != "hang"]
        rods = [o.rod_in * IN for o in opts if o.mode == "hang"]
        out.append((g.id, values[g.id], min(packets) if packets else None, min(rods) if rods else None))
    return out


def knapsack_value(sc: Scenario, agent: str, values: dict[str, float], vol: float, rod: float) -> float:
    items = tuple((gid, v, p, r) for gid, v, p, r in _items(sc, agent, values))
    return _knap(items, round(vol), round(rod * 10))


@lru_cache(maxsize=20000)
def _knap(items: tuple, vol_cap: int, rod_cap_mm: int) -> float:
    m = cp_model.CpModel()
    vol_terms, rod_terms, obj = [], [], []
    for gid, v, p, r in items:
        choices = []
        if p is not None and p <= vol_cap:
            x = m.new_bool_var(f"p{gid}")
            vol_terms.append(int(round(p)) * x)
            choices.append(x)
        if r is not None and r * 10 <= rod_cap_mm:
            y = m.new_bool_var(f"r{gid}")
            rod_terms.append(int(round(r * 10)) * y)
            choices.append(y)
        if choices:
            m.add_at_most_one(choices)
            obj += [int(round(v * 10)) * c for c in choices]
    if not obj:
        return 0.0
    if vol_terms:
        m.add(sum(vol_terms) <= vol_cap)
    if rod_terms:
        m.add(sum(rod_terms) <= rod_cap_mm)
    m.maximize(sum(obj))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = 10
    s.parameters.num_workers = 1
    s.solve(m)
    return s.objective_value / 10.0


def _largest_item(plan: Plan, agent: str) -> tuple[float, float]:
    best = (0.0, 0.0)
    for gid in plan.stored:
        if plan.garments[gid].owner != agent:
            continue
        o = plan.choice[gid]
        item = (0.0, o.rod_in * IN) if o.mode == "hang" else (o.volume, 0.0)
        if item[0] + item[1] * 1000 > best[0] + best[1] * 1000:
            best = item
    return best


def envy_report(sc: Scenario, outcome: Outcome, values: dict[str, float]) -> dict:
    """Average over the lottery of: max envy, number of envious pairs, EF1 violations."""
    agents = sc.agents
    max_envy = pairs = ef1_viol = 0.0
    matrix = {a: {b: 0.0 for b in agents} for a in agents}
    for p, plans in outcome.lottery:
        space = {a: Scenario.space_used(plans[a], a) for a in agents}
        for i in agents:
            own = knapsack_value(sc, i, values, *space[i])
            for j in agents:
                if i == j:
                    continue
                e = max(0.0, knapsack_value(sc, i, values, *space[j]) - own)
                matrix[i][j] += p * e
                if e > 1e-6:
                    pairs += p
                    lv, lr = _largest_item(plans[j], j)
                    vol1, rod1 = max(0.0, space[j][0] - lv), max(0.0, space[j][1] - lr)
                    if knapsack_value(sc, i, values, vol1, rod1) - own > 1e-6:
                        ef1_viol += p
                max_envy = max(max_envy, e)
    return {"max_envy": max_envy, "envious_pairs": pairs, "ef1_violations": ef1_viol, "envy_matrix": matrix}


def proportional_shares(sc: Scenario, values: dict[str, float]) -> dict[str, float]:
    eq = EqualSplit().run(sc, values)
    return {a: sc.stored_value(eq.lottery[0][1][a], a, values) for a in sc.agents}


def fairness(sc: Scenario, outcome: Outcome, values: dict[str, float], ps: dict[str, float] | None = None) -> dict:
    ps = ps or proportional_shares(sc, values)
    stored = {a: sum(p * sc.stored_value(plans[a], a, values) for p, plans in outcome.lottery) for a in sc.agents}
    ratios = {a: (stored[a] / ps[a]) if ps[a] > 0 else float("inf") for a in sc.agents}
    nash = 1.0
    for a in sc.agents:
        nash *= max(stored[a], 0.0)
    env = envy_report(sc, outcome, values)
    return {
        "stored_value": stored,
        "min_share_ratio": min(ratios.values()),
        "below_proportional_share": sum(1 for r in ratios.values() if r < 1 - 1e-9),
        "nash_product": nash,
        **{k: v for k, v in env.items() if k != "envy_matrix"},
        "envy_matrix": env["envy_matrix"],
    }
