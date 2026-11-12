import pytest

from fold_and_place import config
from fold_and_place.folding import generate_options
from fold_and_place.garments import household_garments, make_garment, persona_garments
from fold_and_place.storage import SolveConfig, check, load_preset, solve
from fold_and_place.storage.baselines import BASELINES
from fold_and_place.storage.model import welfare
from fold_and_place.storage.plan import Lane, Plan
from fold_and_place.storage.wardrobe import golden_for

FAST = SolveConfig(time_limit_s=20, workers=4)


@pytest.fixture(scope="module")
def small():
    return persona_garments("mother") + persona_garments("student")[:10]


@pytest.mark.parametrize("preset", list(config.presets()))
def test_every_baseline_produces_a_valid_plan(preset, small):
    w = load_preset(preset)
    for name, fn in BASELINES.items():
        p = fn(small, w)
        assert check(p) == [], (preset, name, check(p)[:3])


def test_optimizer_is_valid_and_at_least_as_good_as_every_baseline(small):
    w = load_preset("one_door")
    opt = solve(small, w, cfg=FAST)
    assert check(opt) == []
    assert opt.status == "OPTIMAL"
    best = max(welfare(fn(small, w)) for fn in BASELINES.values())
    assert welfare(opt) >= best - 1e-6


def test_optimizer_objective_matches_plan_metrics(small):
    w = load_preset("two_door")
    p = solve(small, w, cfg=FAST)
    # the solver maximises welfare + a constant (overflow effort of every garment)
    from fold_and_place.storage.plan import OVERFLOW_REACH, VALUE_WEIGHT

    const = sum(g.wear_freq * OVERFLOW_REACH for g in small)
    assert p.metrics()["objective"] == pytest.approx(welfare(p) - const, abs=0.05)
    assert VALUE_WEIGHT > 0


def test_heavy_lehenga_kept_on_top_and_out_of_loft():
    gs = [make_garment("lehenga", "l1", owner="mother", value=10)] + persona_garments("mother")[:12]
    w = load_preset("two_door")
    p = solve(gs, w, cfg=FAST)
    assert check(p) == []
    lane = p.lane_of("l1")
    assert lane is not None and lane.items[-1] == "l1"
    assert p.compartment_of("l1").kind != "loft"


def test_checker_catches_overfull_stack():
    w = load_preset("one_door")
    g = make_garment("sweater", "s0")
    o = max((o for o in generate_options(g) if o.mode == "flat"), key=lambda o: o.h)
    gs = {f"s{i}": make_garment("sweater", f"s{i}") for i in range(40)}
    choice = {gid: next(x for x in generate_options(gg) if x.method == o.method and x.mode == "flat") for gid, gg in gs.items()}
    plan = Plan(w, gs, choice, [Lane("shelf2", "stack", list(gs))], [])
    errs = check(plan)
    assert any("stack height" in e for e in errs)


def test_checker_catches_double_placement_and_missing():
    w = load_preset("one_door")
    g = make_garment("tshirt", "t1")
    o = generate_options(g)[0]
    lay = "stack" if o.mode == "flat" else ("row" if o.mode in ("file", "roll") else "rail")
    plan = Plan(w, {"t1": g, "t2": make_garment("tshirt", "t2")}, {"t1": o}, [Lane("shelf1", lay, ["t1", "t1"])], [])
    errs = check(plan)
    assert any("twice" in e for e in errs)


def test_values_override_changes_what_is_stored():
    gs = household_garments(scale=3.0)
    w = load_preset("one_door")
    base = solve(gs, w, cfg=FAST)
    if not base.overflow:
        pytest.skip("no scarcity")
    gid = base.overflow[0]
    boosted = solve(gs, w, values={gid: 1000.0}, cfg=FAST)
    assert gid in boosted.stored


def test_golden_zone_uses_owner_height():
    assert golden_for("mother")[1] == 130
    assert golden_for("nobody") == config.golden_zone()


def test_k_loft_override_changes_thickness():
    g = make_garment("jeans", "j-loft")
    base = max(o.h for o in generate_options(g) if o.mode == "flat")
    with config.override_k_loft(config.k_loft() * 1.3):
        thick = max(o.h for o in generate_options(make_garment("jeans", "j-loft2")) if o.mode == "flat")
    assert thick == pytest.approx(base * 1.3, rel=1e-6)


def test_house_rules_are_respected():
    from fold_and_place.garments import household_garments
    from fold_and_place.storage import load_preset
    from fold_and_place.storage.model import SolveConfig, solve
    from fold_and_place.storage.rules import Rules, is_heavy, is_winter

    gs = household_garments()
    w = load_preset("three_door")
    shelves = sorted(c.id for c in w.compartments if c.kind == "shelf")
    rules = Rules(personal={"student": set(shelves[:2])}, pack_winter=True, heavy_low=True)
    p = solve(gs, w, cfg=SolveConfig(time_limit_s=20, workers=2), allowed=rules.allowed(gs, w))
    others = set().union(*rules.personal.values())
    for gid in p.stored:
        g, c = p.garments[gid], p.compartment_of(gid)
        if is_winter(g):
            assert c.kind == "loft" or c.zone_cost() > 1.0
        if is_heavy(g):
            assert c.kind != "loft"
        if g.owner != "student":
            assert c.id not in others
