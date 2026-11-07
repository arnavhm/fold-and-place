"""Strategic behaviour: random misreport search (truthfulness tests) and best-response dynamics
(price of anarchy of manipulable mechanisms)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fold_and_place.mechanisms.mechanisms import Mechanism
from fold_and_place.mechanisms.scenario import Scenario, quantize

MULTIPLIERS = (0.0, 0.5, 1.5, 2.0, 3.0)
GAIN_TOL = 1e-6


@dataclass
class Lie:
    agent: str
    kind: str
    reports: dict[str, float]


def random_lie(sc: Scenario, rng: np.random.Generator) -> Lie:
    """One random deviation by one agent (everyone else truthful):
    scale  - all of the agent's values multiplied by one factor
    mixed  - each of the agent's values multiplied by its own random factor
    single - one garment's value changed (the single-parameter deviation Myerson protects against)
    """
    true = sc.true_values()
    agent = sc.agents[int(rng.integers(len(sc.agents)))]
    mine = [g.id for g in sc.owned(agent)]
    kind = ["scale", "mixed", "single"][int(rng.integers(3))]
    r = dict(true)
    if kind == "scale":
        f = MULTIPLIERS[int(rng.integers(len(MULTIPLIERS)))]
        for gid in mine:
            r[gid] = quantize(true[gid] * f)
    elif kind == "mixed":
        for gid in mine:
            r[gid] = quantize(true[gid] * MULTIPLIERS[int(rng.integers(len(MULTIPLIERS)))])
    else:
        gid = mine[int(rng.integers(len(mine)))]
        r[gid] = quantize(true[gid] * MULTIPLIERS[int(rng.integers(len(MULTIPLIERS)))])
    return Lie(agent, kind, r)


def misreport_test(mech: Mechanism, sc: Scenario, trials: int, seed: int = 0) -> dict:
    """Search for profitable lies. Returns the rate and the largest gain found, by lie kind."""
    rng = np.random.default_rng(seed)
    true = sc.true_values()
    honest = {a: mech.utility(sc, true, a, true) for a in sc.agents}
    found = {"scale": [0, 0], "mixed": [0, 0], "single": [0, 0]}
    max_gain = 0.0
    example = None
    for _ in range(trials):
        lie = random_lie(sc, rng)
        u = mech.utility(sc, lie.reports, lie.agent, true)
        gain = u - honest[lie.agent]
        found[lie.kind][1] += 1
        if gain > GAIN_TOL:
            found[lie.kind][0] += 1
            if gain > max_gain:
                max_gain, example = gain, lie
    profitable = sum(v[0] for v in found.values())
    return {
        "mechanism": mech.name,
        "trials": trials,
        "profitable": profitable,
        "rate": profitable / max(1, trials),
        "max_gain_points": round(max_gain, 3),
        "by_kind": {k: f"{v[0]}/{v[1]}" for k, v in found.items()},
        "example": None if example is None else {"agent": example.agent, "kind": example.kind},
    }


def best_response_dynamics(
    mech: Mechanism, sc: Scenario, multipliers=(0.5, 1.0, 1.5, 2.0, 3.0), rounds: int = 10
) -> dict:
    """Each agent repeatedly picks the uniform report multiplier that maximises its TRUE utility
    given the others' current multipliers, until nobody wants to change (a pure equilibrium in
    this restricted strategy space) or ``rounds`` passes."""
    true = sc.true_values()
    m = {a: 1.0 for a in sc.agents}

    def reports(ms):
        return {g.id: quantize(true[g.id] * ms[g.owner]) for g in sc.garments}

    history = [dict(m)]
    converged = False
    for _ in range(rounds):
        changed = False
        for a in sc.agents:
            # stay put unless some other multiplier is STRICTLY better (ties keep the current one)
            best_f = m[a]
            best_u = mech.utility(sc, reports(m), a, true)
            for f in multipliers:
                if f == m[a]:
                    continue
                trial = {**m, a: f}
                u = mech.utility(sc, reports(trial), a, true)
                if u > best_u + GAIN_TOL:
                    best_f, best_u = f, u
            if best_f != m[a]:
                m[a] = best_f
                changed = True
        history.append(dict(m))
        if not changed:
            converged = True
            break
    out = mech.run(sc, reports(m))
    return {"multipliers": m, "converged": converged, "rounds": len(history) - 1, "outcome": out, "history": history}


def garment_level_test(mech, sc: Scenario, trials: int, seed: int = 0) -> dict:
    """Myerson's promise, tested exactly: each GARMENT is a single-parameter bidder whose utility
    is (its true value if stored) - (its own critical payment). Misreport one garment's value,
    everyone else truthful. Also checks monotonicity: reporting more never un-stores it."""
    rng = np.random.default_rng(seed)
    true = sc.true_values()
    gids = [g.id for g in sc.garments]

    def u(r, gid):
        plan = mech._plan(sc, r)
        if gid not in plan.stored:
            return 0.0
        return true[gid] - mech.critical(sc, r, gid)

    profitable, non_monotone, max_gain = 0, 0, 0.0
    for _ in range(trials):
        gid = gids[int(rng.integers(len(gids)))]
        f = MULTIPLIERS[int(rng.integers(len(MULTIPLIERS)))]
        r = {**true, gid: quantize(true[gid] * f)}
        gain = u(r, gid) - u(true, gid)
        if gain > mech.tol + GAIN_TOL:  # payments are found by bisection to within tol
            profitable += 1
            max_gain = max(max_gain, gain)
        lo, hi = sorted((true[gid], r[gid]))
        if gid in mech._plan(sc, {**true, gid: lo}).stored and gid not in mech._plan(sc, {**true, gid: hi}).stored:
            non_monotone += 1
    return {
        "mechanism": mech.name + " [per garment]",
        "trials": trials,
        "profitable": profitable,
        "rate": profitable / max(1, trials),
        "max_gain_points": round(max_gain, 3),
        "non_monotone": non_monotone,
    }
