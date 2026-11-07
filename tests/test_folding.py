import math

import numpy as np
import pytest

from fold_and_place import config
from fold_and_place.folding import FoldState, FoldStep, apply_fold, generate_options
from fold_and_place.folding.calibration import BOARD_SHIRT_CM, board_check, calibrate_k_loft
from fold_and_place.folding.engine import fit_length, panels, sides_in
from fold_and_place.folding.options import MAX_SIDE_CM
from fold_and_place.garments import make_garment

ALL_CATEGORIES = list(config.categories().keys())


def random_steps(state: FoldState, rng: np.random.Generator, n: int) -> FoldState:
    for _ in range(n):
        axis = "x" if rng.random() < 0.5 else "y"
        x0, y0, x1, y1 = state.bounds
        lo, hi = (x0, x1) if axis == "x" else (y0, y1)
        c = lo + (hi - lo) * rng.uniform(0.15, 0.85)
        state = apply_fold(state, FoldStep(axis, c, "lt" if rng.random() < 0.5 else "gt"))
    return state


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_area_is_conserved_under_any_folds(category):
    g = make_garment(category, "g")
    st0 = FoldState.from_garment(g)
    rng = np.random.default_rng(7)
    st = random_steps(st0, rng, 5)
    assert st.ply_area() == pytest.approx(st0.ply_area(), rel=1e-6)


def test_folds_are_deterministic():
    g = make_garment("shirt", "g", "L")
    a = generate_options(g)
    b = generate_options(make_garment("shirt", "h", "L"))
    assert [(o.mode, round(o.a, 6), round(o.b, 6), round(o.h, 6)) for o in a] == [
        (o.mode, round(o.a, 6), round(o.b, 6), round(o.h, 6)) for o in b
    ]


def test_rect_panels_halve_length_and_double_plies():
    g = make_garment("dupatta", "d")
    st = panels(FoldState.from_garment(g), "y", 4)
    assert st.length == pytest.approx(g.flat_l / 4)
    assert st.max_plies() == 4 * g.base_layers


def test_sides_in_leaves_exact_width_even_with_long_sleeves():
    g = make_garment("shirt", "s", "L")  # 60 cm sleeves stick out far past the panel
    st = sides_in(FoldState.from_garment(g), 23.0)
    assert st.width == pytest.approx(23.0, abs=1e-6)


def test_fit_length_hits_target():
    g = make_garment("kurta_men", "k")
    st = fit_length(sides_in(FoldState.from_garment(g), 23.0), 30.5)
    assert st.length == pytest.approx(30.5, abs=1e-6)


def test_board_fold_matches_department_store_board():
    rows = board_check("shirt", "L")
    board = next(r for r in rows if r["method"].startswith("board"))
    bl, bw = BOARD_SHIRT_CM
    assert board["folded_w_cm"] == pytest.approx(bw, rel=0.10)
    assert board["folded_l_cm"] == pytest.approx(bl, rel=0.10)


def test_config_loft_factor_matches_calibration():
    """configs/fabrics.yaml must hold the value E1 calibrates (rounded to 2 dp)."""
    assert config.k_loft() == pytest.approx(calibrate_k_loft()["k_loft"], abs=0.01)
    assert 1.0 <= config.k_loft() <= 2.5  # air between layers: never thinner than the plies


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_every_category_has_options_that_fit(category):
    g = make_garment(category, "g")
    opts = generate_options(g)
    assert opts, category
    for o in opts:
        if o.mode != "hang":
            assert o.a <= MAX_SIDE_CM + 1e-6
            assert o.b <= o.a + 1e-9
            assert o.h > 0
            assert o.creases == len(o.steps)


def test_heavy_embellished_pieces_are_folded_flat_only():
    for cat in ("lehenga", "sherwani"):
        modes = {o.mode for o in generate_options(make_garment(cat, "x"))}
        assert modes == {"flat"}, (cat, modes)


def test_hang_only_and_no_hang_rules():
    assert {o.mode for o in generate_options(make_garment("blazer", "b"))} == {"hang"}
    assert "hang" not in {o.mode for o in generate_options(make_garment("socks", "s"))}


def test_roll_diameter_grows_with_length():
    short = [o for o in generate_options(make_garment("tshirt", "a", "S")) if o.mode == "roll"]
    long_ = [o for o in generate_options(make_garment("kurta_men", "b", "L")) if o.mode == "roll"]
    assert short and long_
    assert max(o.h for o in long_) > min(o.h for o in short)


def test_option_ids_are_per_garment():
    a = generate_options(make_garment("jeans", "alice-jeans"))
    assert all(o.id.startswith("alice-jeans#") and o.garment_id == "alice-jeans" for o in a)
    assert len({o.id for o in a}) == len(a)


def test_department_store_option_exists_for_tops():
    opts = generate_options(make_garment("tshirt", "t", "L"))
    ds = [o for o in opts if "Department-store" in o.method]
    assert ds and math.isclose(ds[0].a, 30.5, rel_tol=0.01)
