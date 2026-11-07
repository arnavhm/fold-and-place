"""Who should pay for the almirah? A cooperative cost game.

c(S) = price of the cheapest preset that stores ALL garments of coalition S (an airport-game
cousin: bigger groups need bigger wardrobes, but sharing one is cheaper than separate ones).
Feasibility: any baseline plan that stores everything, else the CP-SAT optimizer; a coalition
the optimizer cannot fit within its time limit is treated as not fitting (conservative).
"""

from __future__ import annotations

import itertools
import math

from fold_and_place import config
from fold_and_place.garments import persona_garments
from fold_and_place.garments.record import Garment
from fold_and_place.storage import SolveConfig, load_preset, solve
from fold_and_place.storage.baselines import BASELINES


def fits_all(gs: list[Garment], preset: str, time_limit: float = 30.0) -> bool:
    w = load_preset(preset)
    for fn in BASELINES.values():
        if not fn(gs, w).overflow:
            return True
    return not solve(gs, w, cfg=SolveConfig(time_limit_s=time_limit)).overflow


def cost_function(agents: list[str], scale: float, time_limit: float = 30.0) -> dict[frozenset, tuple[float, str]]:
    prices = config.load("presets")["prices_inr"]
    order = sorted(prices, key=prices.get)
    owned = {a: persona_garments(a, scale=scale) for a in agents}
    c: dict[frozenset, tuple[float, str]] = {frozenset(): (0.0, "-")}
    for k in range(1, len(agents) + 1):
        for S in itertools.combinations(agents, k):
            gs = [g for a in S for g in owned[a]]
            choice = (math.inf, "none fits")
            for p in order:
                if fits_all(gs, p, time_limit):
                    choice = (float(prices[p]), p)
                    break
            c[frozenset(S)] = choice
    return c


def shapley(agents: list[str], c: dict[frozenset, tuple[float, str]]) -> dict[str, float]:
    n = len(agents)
    phi = {}
    for i in agents:
        rest = [a for a in agents if a != i]
        tot = 0.0
        for k in range(len(rest) + 1):
            for S in itertools.combinations(rest, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                tot += w * (c[frozenset(S) | {i}][0] - c[frozenset(S)][0])
        phi[i] = tot
    return phi


def in_core(agents: list[str], c, shares: dict[str, float]) -> tuple[bool, list[str]]:
    """Cost-game core: no group pays more in total than buying its own wardrobe would cost."""
    bad = []
    for k in range(1, len(agents)):
        for S in itertools.combinations(agents, k):
            if sum(shares[a] for a in S) > c[frozenset(S)][0] + 1e-6:
                bad.append(" + ".join(S))
    return (not bad), bad
