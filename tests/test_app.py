"""Dashboard: the 3D fold geometry matches the folding engine, and every page renders."""

from __future__ import annotations

import numpy as np
import pytest

from fold_and_place.folding.engine import run
from fold_and_place.folding.options import generate_options
from fold_and_place.garments.record import make_garment

fold3d = pytest.importorskip("fold_and_place.app.fold3d")


def _final_planar(d):
    V = np.array(d["verts"]).reshape(-1, 2)
    for st in d["steps"]:
        if not st["mask"]:
            continue
        m = np.array([c == "1" for c in st["mask"]])
        a = 0 if st["axis"] == "x" else 1
        V[m, a] = 2 * st["c"] - V[m, a]
    return V


@pytest.mark.parametrize("cat", ["tshirt", "shirt", "jeans", "saree", "lehenga", "dress", "dupatta"])
def test_animation_footprint_matches_engine(cat):
    g = make_garment(cat, "x")
    for o in [o for o in generate_options(g) if o.steps][:3]:
        d = fold3d.build(g, o.steps)
        V = _final_planar(d)
        x0, y0, x1, y1 = run(g, o.steps).bounds
        assert np.allclose([V[:, 0].min(), V[:, 1].min(), V[:, 0].max(), V[:, 1].max()], [x0, y0, x1, y1], atol=1e-3)


def test_mesh_covers_outline_area():
    from fold_and_place.folding.templates import outline

    g = make_garment("lehenga", "x")
    d = fold3d.build(g, [])
    V = np.array(d["verts"]).reshape(-1, 2)
    T = np.array(d["tris"]).reshape(-1, 3)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    signed = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    assert (signed > 0).all()  # all counter-clockwise: the front face is up
    assert abs(signed.sum() - outline(g).area) / outline(g).area < 1e-6


def test_accordion_stack_grows_linearly():
    g = make_garment("saree", "x")
    o = next(o for o in generate_options(g) if o.mode == "flat" and "10 panels" in o.method and "in 2" in o.method)
    d = fold3d.build(g, o.steps)
    # 1 ply x 2 (lengthwise) x 10 panels = 20 layers -> 20 gaps, not 2^10
    assert d["final_h"] == pytest.approx(20 * fold3d.GAP_CM, abs=1e-6)


def test_html_is_self_contained():
    g = make_garment("tshirt", "x")
    o = generate_options(g)[0]
    page = fold3d.html(g, o.steps, done="ok")
    assert "THREE.WebGLRenderer" in page and "OrbitControls" in page and "/*DATA*/" not in page
    assert "http://" not in page.split("<script>", 1)[0]  # no external resources in the head


PAGES = ["views/wardrobe.py", "views/clothes.py", "views/plan.py", "views/sharing.py", "views/fold_studio.py",
         "views/experiments.py", "views/method.py"]


def test_every_page_renders():
    testing = pytest.importorskip("streamlit.testing.v1")
    from fold_and_place.config import ROOT
    from fold_and_place.storage import wardrobe

    try:
        at = testing.AppTest.from_file(str(ROOT / "fold_and_place" / "app" / "app.py"), default_timeout=240).run()
        assert not at.exception, [e.value for e in at.exception]
        for page in PAGES:
            at.switch_page(page).run()
            assert not at.exception, (page, [e.value for e in at.exception])
    finally:
        wardrobe.SHOULDER_CM.clear()  # the app sets people's heights process-wide


def test_printable_plan_lists_every_stored_garment():
    from fold_and_place.app.printable import plan_html
    from fold_and_place.garments import household_garments
    from fold_and_place.storage import load_preset
    from fold_and_place.storage.baselines import b1_traditional

    gs = household_garments()[:30]
    p = b1_traditional(gs, load_preset("two_door"))
    page = plan_html(p)
    assert page.count("<li><b>") == len(p.stored)
    assert "<svg" in page
