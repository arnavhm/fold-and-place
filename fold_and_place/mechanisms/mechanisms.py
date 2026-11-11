"""Mechanisms for sharing one wardrobe among people with private values.

Every mechanism maps REPORTED values (garment id -> points) to an outcome: one or more plans (a
lottery for RSD) and payments in "space credits" (value points; positive = the person pays).

  VCG            exact welfare optimum + Clarke pivot payments          truthful, efficient
  VCG+Bailey     VCG, then 1/n of VCG revenue without you, paid back     truthful, closer to balance
  Myerson        garment-level knapsack auction: greedy (monotone) allocation + critical-bid payments
  RSD            random serial dictatorship (no money)                  strategyproof, ex-post efficient
  MNW            maximum Nash welfare (no money)                        fair (EF1 under no constraints)
  Equal split    each person gets 1/n of every shelf/drawer/rail
  FCFS           garments placed in a fixed arrival order
  Proportional   space split by the share of value each person CLAIMS   manipulable
  Priority       greedy by reported value, no payments                  manipulable
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from fold_and_place.mechanisms.scenario import Scenario, quantize, shrink
from fold_and_place.storage.baselines import b4_greedy
from fold_and_place.storage.plan import Plan


@dataclass
class Outcome:
    mechanism: str
    lottery: list[tuple[float, dict[str, Plan]]]  # (probability, plan each agent's garments live in)
    payments: dict[str, float] = field(default_factory=dict)
    rebates: dict[str, float] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def value(self, sc: Scenario, agent: str, values: dict[str, float]) -> float:
        return sum(p * sc.v(plans[agent], agent, values) for p, plans in self.lottery)

    def utility(self, sc: Scenario, agent: str, values: dict[str, float]) -> float:
        return self.value(sc, agent, values) - self.payments.get(agent, 0.0) + self.rebates.get(agent, 0.0)

    def welfare(self, sc: Scenario, values: dict[str, float]) -> float:
        return sum(self.value(sc, a, values) for a in sc.agents)

    def plan(self) -> Plan | None:
        """The plan to show (most likely one); None if every agent has a separate share."""
        _, plans = max(self.lottery, key=lambda t: t[0])
        ps = list({id(p): p for p in plans.values()}.values())
        return ps[0] if len(ps) == 1 else None


def _joint(sc: Scenario, plan: Plan) -> list[tuple[float, dict[str, Plan]]]:
    return [(1.0, {a: plan for a in sc.agents})]


def _clean(reports: dict[str, float]) -> dict[str, float]:
    return {k: quantize(v) for k, v in reports.items()}


class Mechanism:
    name = "mechanism"
    uses_money = False
    claim = ""

    def run(self, sc: Scenario, reports: dict[str, float]) -> Outcome:  # pragma: no cover
        raise NotImplementedError

    def utility(self, sc: Scenario, reports: dict[str, float], agent: str, true: dict[str, float]) -> float:
        """Agent's TRUE utility when everyone reports ``reports``. Overridden where a faster
        computation exists (only the parts that depend on ``agent``)."""
        return self.run(sc, reports).utility(sc, agent, true)


# --------------------------------------------------------------------------- with money
class VCG(Mechanism):
    name = "VCG"
    uses_money = True
    claim = "truthful (dominant strategy), welfare-maximising; not budget-balanced"

    def payment(self, sc: Scenario, r: dict[str, float], plan: Plan, agent: str) -> float:
        others = [a for a in sc.agents if a != agent]
        h = sc.best_welfare(r, others)  # others' best welfare if this agent were not there
        return h - sum(sc.v(plan, a, r) for a in others)

    def run(self, sc: Scenario, reports: dict[str, float]) -> Outcome:
        r = _clean(reports)
        plan = sc.optimum(r)
        pay = {a: self.payment(sc, r, plan, a) for a in sc.agents}
        return Outcome(self.name, _joint(sc, plan), payments=pay, info={"status": plan.status})

    def utility(self, sc, reports, agent, true):
        r = _clean(reports)
        plan = sc.optimum(r)
        return sc.v(plan, agent, true) - self.payment(sc, r, plan, agent)


class VCGBailey(VCG):
    name = "VCG + Bailey redistribution"
    claim = "truthful and efficient like VCG; pays back part of the revenue"

    def revenue_without(self, sc: Scenario, r: dict[str, float], agent: str) -> float:
        """Total VCG payments in the economy without ``agent`` (depends only on others' reports)."""
        rest = [a for a in sc.agents if a != agent]
        if len(rest) < 2:
            return 0.0
        plan = sc.optimum(r, rest)
        total = 0.0
        for j in rest:
            others = [a for a in rest if a != j]
            total += sc.best_welfare(r, others) - sum(sc.v(plan, a, r) for a in others)
        return total

    def rebate(self, sc: Scenario, r: dict[str, float], agent: str) -> float:
        return self.revenue_without(sc, r, agent) / len(sc.agents)

    def run(self, sc, reports):
        out = super().run(sc, reports)
        r = _clean(reports)
        out.mechanism = self.name
        out.rebates = {a: self.rebate(sc, r, a) for a in sc.agents}
        return out

    def utility(self, sc, reports, agent, true):
        r = _clean(reports)
        return super().utility(sc, reports, agent, true) + self.rebate(sc, r, agent)


class Myerson(Mechanism):
    """Each garment is a single-parameter bidder (public size, private value). Allocation: greedy
    by value then value density (monotone); each stored garment pays its CRITICAL value - the
    lowest report at which it would still have been stored (Myerson's lemma)."""

    name = "Myerson knapsack auction (greedy)"
    uses_money = True
    claim = "truthful per garment if the greedy rule is monotone (tested); ~1/2-approx welfare"
    tol = 0.05

    def _plan(self, sc: Scenario, r: dict[str, float]) -> Plan:
        key = ("myerson", tuple(sorted(r.items())))
        if key not in sc._cache:
            sc._cache[key] = b4_greedy(
                sc.garments, sc.wardrobe, values=r, lam_reach=sc.cfg.lam_reach, lam_crease=sc.cfg.lam_crease, prefer="space"
            )
        return sc._cache[key]

    def critical(self, sc: Scenario, r: dict[str, float], gid: str) -> float:
        lo, hi = 0.0, r[gid]
        if gid in self._plan(sc, {**r, gid: 0.0}).stored:
            return 0.0
        while hi - lo > self.tol:
            mid = (lo + hi) / 2
            if gid in self._plan(sc, {**r, gid: mid}).stored:
                hi = mid
            else:
                lo = mid
        return hi

    def payment(self, sc: Scenario, r: dict[str, float], plan: Plan, agent: str) -> float:
        return sum(self.critical(sc, r, gid) for gid in plan.stored if plan.garments[gid].owner == agent)

    def run(self, sc, reports):
        r = dict(reports)
        plan = self._plan(sc, r)
        return Outcome(self.name, _joint(sc, plan), payments={a: self.payment(sc, r, plan, a) for a in sc.agents})

    def utility(self, sc, reports, agent, true):
        r = dict(reports)
        plan = self._plan(sc, r)
        return sc.v(plan, agent, true) - self.payment(sc, r, plan, agent)


# --------------------------------------------------------------------------- no money
class RSD(Mechanism):
    """Random serial dictatorship: a random order; each person in turn stores what is best for
    them in the space the earlier people left. Computed exactly with sequential solves (each
    earlier person's utility is held as a constraint), averaged over all n! orders."""

    name = "Random serial dictatorship"
    claim = "strategyproof, ex-post Pareto efficient, equal treatment; not envy-free"

    def run(self, sc, reports):
        r = _clean(reports)
        orders = list(itertools.permutations(sc.agents))
        lottery = [(1.0 / len(orders), {a: sc.dictatorship(r, o) for a in sc.agents}) for o in orders]
        return Outcome(self.name, lottery, info={"orders": len(orders)})


class MNW(Mechanism):
    name = "Maximum Nash welfare"
    claim = "EF1 + Pareto with unconstrained goods; under space limits EF1 is only approximate (measured)"

    def run(self, sc, reports):
        r = _clean(reports)
        plan = sc.optimum(r, objective="nash")
        return Outcome(self.name, _joint(sc, plan), info={"status": plan.status})


class EqualSplit(Mechanism):
    name = "Equal split"
    claim = "each person optimises their own 1/n of every compartment; truthful, wasteful"

    def share_plan(self, sc: Scenario, r: dict[str, float], agent: str, factor: float, tag: str) -> Plan:
        return sc.optimum(r, [agent], shrink(sc.wardrobe, factor, tag))

    def run(self, sc, reports):
        r = _clean(reports)
        n = len(sc.agents)
        plans = {a: self.share_plan(sc, r, a, 1.0 / n, "eq") for a in sc.agents}
        return Outcome(self.name, [(1.0, plans)])

    def utility(self, sc, reports, agent, true):
        r = _clean(reports)
        return sc.v(self.share_plan(sc, r, agent, 1.0 / len(sc.agents), "eq"), agent, true)


class Proportional(EqualSplit):
    name = "Proportional to claims"
    claim = "space split by each person's share of claimed value; manipulable (inflate claims)"

    def shares(self, sc: Scenario, r: dict[str, float]) -> dict[str, float]:
        tot = {a: sum(r[g.id] for g in sc.owned(a)) for a in sc.agents}
        s = sum(tot.values()) or 1.0
        return {a: tot[a] / s for a in sc.agents}

    def run(self, sc, reports):
        r = _clean(reports)
        sh = self.shares(sc, r)
        plans = {a: self.share_plan(sc, r, a, max(sh[a], 1e-6), "prop") for a in sc.agents}
        return Outcome(self.name, [(1.0, plans)], info={"shares": sh})

    def utility(self, sc, reports, agent, true):
        r = _clean(reports)
        sh = self.shares(sc, r)
        return sc.v(self.share_plan(sc, r, agent, max(sh[agent], 1e-6), "prop"), agent, true)


class FCFS(Mechanism):
    name = "First come, first served"
    claim = "garments placed in a fixed arrival order; ignores values (truthful, wasteful)"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def run(self, sc, reports):
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(sc.garments))
        arrival = {g.id: float(len(order) - int(i)) for g, i in zip(sc.garments, order)}
        key = ("fcfs", self.seed)
        if key not in sc._cache:
            sc._cache[key] = b4_greedy(sc.garments, sc.wardrobe, values=arrival, prefer="space")
        return Outcome(self.name, _joint(sc, sc._cache[key]))


class Priority(Mechanism):
    name = "Priority by claimed value"
    claim = "greedy by reported value, no payments; manipulable (inflate claims)"

    def run(self, sc, reports):
        r = _clean(reports)
        key = ("priority", tuple(sorted(r.items())))
        if key not in sc._cache:
            sc._cache[key] = b4_greedy(
                sc.garments, sc.wardrobe, values=r, lam_reach=sc.cfg.lam_reach, lam_crease=sc.cfg.lam_crease, prefer="space"
            )
        return Outcome(self.name, _joint(sc, sc._cache[key]))


ALL = [VCG(), VCGBailey(), Myerson(), RSD(), MNW(), EqualSplit(), FCFS(), Proportional(), Priority()]
BY_NAME = {m.name: m for m in ALL}


def run_all(sc: Scenario, reports: dict[str, float] | None = None, which=None) -> dict[str, Outcome]:
    r = reports or sc.true_values()
    return {m.name: m.run(sc, r) for m in (which or ALL)}


def nash_product(vals: list[float]) -> float:
    return math.prod(max(v, 0.0) for v in vals)
