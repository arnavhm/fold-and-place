"""A shared-wardrobe allocation problem: who (agent) owns which garments, the wardrobe, and an
exact allocation oracle (the CP-SAT optimizer) with a cache.

Units: everything here is in VALUE POINTS. An agent's valuation of a plan is

    v_i(plan) = sum over i's stored garments of  value_g + (reach saving - crease cost) / 20

i.e. one point = one unit of garment value; the reach/crease terms are the optimizer's own terms
divided by its value weight (20). Private information = the values; sizes, wear frequency and
fabric are public (vision measures them).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from fold_and_place.garments import household_garments
from fold_and_place.garments.record import Garment
from fold_and_place.storage import SolveConfig, load_preset, solve
from fold_and_place.storage.plan import IN, OVERFLOW_REACH, VALUE_WEIGHT, Plan, reach_cost
from fold_and_place.storage.wardrobe import Wardrobe

REPORT_STEP = 0.5  # reports are rounded to half points (keeps lexicographic solves exact)


def quantize(v: float) -> float:
    return max(0.0, round(v / REPORT_STEP) * REPORT_STEP)


def shrink(w: Wardrobe, factor: float, tag: str = "") -> Wardrobe:
    """A copy of ``w`` whose compartments are ``factor`` as wide (a share of every shelf, drawer
    and rail): what one agent gets under an equal or proportional split."""
    comps = [dataclasses.replace(c, w=c.w * factor) for c in w.compartments]
    return Wardrobe(f"{w.name}*{factor:.4f}{tag}", w.label, w.width * factor, w.height, w.depth, comps)


@dataclass
class Scenario:
    garments: list[Garment]
    wardrobe: Wardrobe
    # One solver worker with a fixed seed: when several plans tie, the same one is picked every
    # run, so payments do not shift between runs (utilities never depended on it).
    cfg: SolveConfig = field(default_factory=lambda: SolveConfig(time_limit_s=60, workers=1, warm_start=False))
    _cache: dict = field(default_factory=dict, repr=False)
    solves: int = 0

    @classmethod
    def default(cls, scale: float = 0.4, preset: str = "compact_shared", seed: int = 0) -> Scenario:
        return cls(household_garments(scale=scale, seed=seed), load_preset(preset))

    # ----- basic facts ------------------------------------------------------------------
    @property
    def agents(self) -> list[str]:
        return sorted({g.owner for g in self.garments})

    @property
    def by_id(self) -> dict[str, Garment]:
        return {g.id: g for g in self.garments}

    def owned(self, agent: str) -> list[Garment]:
        return [g for g in self.garments if g.owner == agent]

    def true_values(self) -> dict[str, float]:
        return {g.id: float(g.value) for g in self.garments}

    # ----- valuations ------------------------------------------------------------------------
    def extra(self, plan: Plan, gid: str) -> float:
        """Reach saving minus crease cost of a stored garment, in value points."""
        g, o = plan.garments[gid], plan.choice[gid]
        c = plan.compartment_of(gid)
        cfg = self.cfg
        return (
            cfg.lam_reach * g.wear_freq * (OVERFLOW_REACH - reach_cost(g, o, c))
            - cfg.lam_crease * g.crease_risk * o.creases
        ) / VALUE_WEIGHT

    def v(self, plan: Plan, agent: str, values: dict[str, float]) -> float:
        return sum(
            values[gid] + self.extra(plan, gid) for gid in plan.stored if plan.garments[gid].owner == agent
        )

    def stored_value(self, plan: Plan, agent: str, values: dict[str, float]) -> float:
        """Value only (no reach/crease terms) - used for fairness measures."""
        return sum(values[gid] for gid in plan.stored if plan.garments[gid].owner == agent)

    def welfare(self, plan: Plan, values: dict[str, float], agents: list[str] | None = None) -> float:
        return sum(self.v(plan, a, values) for a in (agents or self.agents))

    # ----- the allocation oracle (exact optimum, cached) ---------------------------------
    def optimum(
        self,
        values: dict[str, float],
        agents: list[str] | None = None,
        wardrobe: Wardrobe | None = None,
        objective: str = "welfare",
    ) -> Plan:
        agents = agents if agents is not None else self.agents
        gs = [g for g in self.garments if g.owner in agents]
        w = wardrobe or self.wardrobe
        key = (
            tuple(g.id for g in gs),
            tuple(round(values[g.id], 6) for g in gs),
            w.name,
            tuple((c.id, round(c.w, 4)) for c in w.compartments),
            objective,
        )
        if key not in self._cache:
            self.solves += 1
            self._cache[key] = solve(gs, w, values={g.id: values[g.id] for g in gs}, cfg=self.cfg, objective=objective)
        return self._cache[key]

    def dictatorship(self, values: dict[str, float], order: tuple[str, ...]) -> Plan:
        """Serial dictatorship, exactly: each person in ``order`` maximises their own utility
        while everyone before them keeps exactly what they already secured."""
        floors: dict[str, int] = {}
        plan = None
        for k, who in enumerate(order):
            key = ("sd", tuple(round(values[g.id], 6) for g in self.garments), order[: k + 1], self.wardrobe.name)
            if key not in self._cache:
                self.solves += 1
                self._cache[key] = solve(
                    self.garments, self.wardrobe, values=values, cfg=self.cfg, objective="owner", owner=who, floors=dict(floors)
                )
            plan = self._cache[key]
            floors[who] = plan.owner_scores[who]
        return plan

    def best_welfare(self, values: dict[str, float], agents: list[str]) -> float:
        """max over plans of the welfare of ``agents`` when only their garments compete."""
        if not agents:
            return 0.0
        return self.welfare(self.optimum(values, agents), values, agents)

    # ----- space used (for envy) -----------------------------------------------------------
    @staticmethod
    def space_used(plan: Plan, agent: str) -> tuple[float, float]:
        """(packet volume cm^3, rod cm) used by ``agent``'s stored garments."""
        vol = rod = 0.0
        for gid in plan.stored:
            if plan.garments[gid].owner != agent:
                continue
            o = plan.choice[gid]
            if o.mode == "hang":
                rod += o.rod_in * IN
            else:
                vol += o.volume
        return vol, rod
