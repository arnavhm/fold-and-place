import itertools

import pytest

from fold_and_place.mechanisms import Scenario
from fold_and_place.mechanisms.fairness import fairness, proportional_shares
from fold_and_place.mechanisms.mechanisms import (
    MNW,
    RSD,
    VCG,
    EqualSplit,
    Myerson,
    Priority,
    Proportional,
    VCGBailey,
)
from fold_and_place.mechanisms.shapley import in_core, shapley
from fold_and_place.mechanisms.strategic import garment_level_test, misreport_test
from fold_and_place.storage import check


@pytest.fixture(scope="module")
def sc():
    s = Scenario.default(scale=0.3)
    s.cfg.workers = 2
    return s


@pytest.fixture(scope="module")
def true(sc):
    return sc.true_values()


def test_scenario_is_scarce(sc, true):
    plan = sc.optimum(true)
    assert check(plan) == []
    assert plan.overflow, "the mechanism scenario must leave something out"


def test_vcg_payments_are_clarke_and_individually_rational(sc, true):
    out = VCG().run(sc, true)
    for a in sc.agents:
        assert out.payments[a] >= -1e-6  # Clarke pivot: never paid to participate
        assert out.utility(sc, a, true) >= -1e-6  # truthful participation never hurts
    assert out.welfare(sc, true) == pytest.approx(sc.welfare(sc.optimum(true), true))


def test_vcg_no_profitable_lie(sc):
    res = misreport_test(VCG(), sc, trials=12, seed=3)
    assert res["profitable"] == 0, res


def test_bailey_rebates_keep_incentives_and_reduce_surplus(sc, true):
    vcg = VCG().run(sc, true)
    bai = VCGBailey().run(sc, true)
    assert all(r >= -1e-6 for r in bai.rebates.values())
    assert sum(bai.payments.values()) - sum(bai.rebates.values()) <= sum(vcg.payments.values()) + 1e-6
    assert misreport_test(VCGBailey(), sc, trials=6, seed=4)["profitable"] == 0


def test_rsd_is_a_lottery_and_strategyproof(sc, true):
    out = RSD().run(sc, true)
    assert sum(p for p, _ in out.lottery) == pytest.approx(1.0)
    assert len(out.lottery) == len(list(itertools.permutations(sc.agents)))
    assert misreport_test(RSD(), sc, trials=6, seed=5)["profitable"] == 0


def test_myerson_is_truthful_and_monotone_per_garment(sc, true):
    res = garment_level_test(Myerson(), sc, trials=60, seed=6)
    assert res["profitable"] == 0 and res["non_monotone"] == 0, res
    out = Myerson().run(sc, true)
    plan = out.plan()
    for a in sc.agents:  # nobody pays more than they reported for what they got stored
        assert out.payments[a] <= sum(true[g] for g in plan.stored if plan.garments[g].owner == a) + 1e-6


def test_equal_split_gives_everyone_exactly_their_proportional_share(sc, true):
    ps = proportional_shares(sc, true)
    f = fairness(sc, EqualSplit().run(sc, true), true, ps)
    assert f["below_proportional_share"] == 0
    assert f["min_share_ratio"] == pytest.approx(1.0)


def test_naive_rules_are_manipulable(sc):
    """Inflating your claims pays under Priority and Proportional (that is why we need mechanisms)."""
    true = sc.true_values()
    for mech in (Priority(), Proportional()):
        gains = []
        for a in sc.agents:
            honest = mech.utility(sc, true, a, true)
            lie = {g.id: (true[g.id] * 3 if g.owner == a else true[g.id]) for g in sc.garments}
            gains.append(mech.utility(sc, lie, a, true) - honest)
        assert max(gains) > 0, mech.name


def test_mnw_maximises_the_product(sc, true):
    mnw = fairness(sc, MNW().run(sc, true), true)
    vcg = fairness(sc, VCG().run(sc, true), true)
    assert mnw["nash_product"] >= vcg["nash_product"] - 1e-6


def test_shapley_airport_game_matches_textbook():
    """Wikipedia 'Airport problem': runway costs 8, 11, 13, 18 -> Shapley 2, 3, 4, 9."""
    costs = {"A": 8, "B": 11, "C": 13, "D": 18}
    agents = list(costs)
    c = {frozenset(): (0.0, "-")}
    for k in range(1, 5):
        for S in itertools.combinations(agents, k):
            c[frozenset(S)] = (float(max(costs[a] for a in S)), "")
    phi = shapley(agents, c)
    assert [round(phi[a], 6) for a in agents] == [2, 3, 4, 9]
    assert sum(phi.values()) == pytest.approx(18)
    assert in_core(agents, c, phi)[0]
    assert not in_core(agents, c, {"A": 18, "B": 0, "C": 0, "D": 0})[0]
