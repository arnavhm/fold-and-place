"""The real-photo path on realistic synthetic photos: textured bedsheet, prints, creases,
shadows, uneven light, any rotation, up to 25 degrees of camera tilt, JPEG noise.
Needs OpenCV (installed with the vision extras); skipped otherwise."""

import numpy as np
import pytest

from fold_and_place.config import categories
from fold_and_place.folding.engine import run
from fold_and_place.folding.options import generate_options
from fold_and_place.vision.measure import drop_marker, homography
from fold_and_place.vision.models import PlainSegmenter
from fold_and_place.vision.shape import mirror_iou, silhouette

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from fold_and_place.vision.reference import detect_a4, main_part  # noqa: E402
from fold_and_place.vision.synth import render  # noqa: E402
from tests.test_vision import fake  # noqa: E402

CASES = [("tshirt", 1), ("jeans", 2), ("kurti", 5), ("skirt", 14), ("trousers", 13), ("dress", 18), ("palazzo", 9)]


@pytest.mark.parametrize("cat,seed", CASES)
def test_a4_found_and_garment_measured(cat, seed):
    ph = render(cat, "M", seed=seed)
    px, cm = detect_a4(ph.image)
    err = min(np.abs(np.roll(px, k, 0) - ph.sheet_px).max() for k in range(4))
    assert err < 4.0  # pixels
    mask = main_part(drop_marker(PlainSegmenter()(ph.image), px, grow_px=8))
    s = silhouette(mask, ph.garment.template, homography(px, cm))
    g = ph.garment
    assert abs(s.measured.flat_l - g.flat_l) / g.flat_l < 0.08
    assert abs(s.measured.flat_w - g.flat_w) / g.flat_w < 0.12


def test_no_sheet_no_false_reference():
    ph = render("tshirt", "M", seed=3, sheet=False)
    assert detect_a4(ph.image) is None


@pytest.mark.parametrize("k", range(4))
def test_any_rotation_is_turned_upright(k):
    from matplotlib.path import Path as MplPath

    from fold_and_place.folding.templates import outline
    from fold_and_place.garments import make_garment

    g = make_garment("jeans", "x")
    P = np.asarray(outline(g).exterior.coords)[:-1]
    P = np.c_[P[:, 0], -P[:, 1]]
    th = k * np.pi / 2 + 0.3
    P = P @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).T
    Q = (P - P.min(0)) * 5 + 10
    W, H = (Q.max(0) + 10).astype(int)
    yy, xx = np.mgrid[0:H, 0:W]
    mask = MplPath(Q).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(H, W)
    s = silhouette(mask, "trousers", np.diag([0.2, 0.2, 1]))
    assert abs(s.measured.flat_l - 104) < 2 and abs(s.measured.flat_w - 50) < 2
    # legs at the bottom of the upright grid
    bottom = s.grid[-5]
    assert ((np.diff(np.r_[0, bottom.astype(int)]) == 1).sum()) == 2
    assert mirror_iou(s.grid) > 0.9


def test_pipeline_finds_sheet_measures_and_keeps_outline():
    ph = render("tshirt", "M", seed=10)
    guess = fake("tshirt", fabric="cotton_jersey").analyse(ph.image, size="M")
    assert guess.size_source == "marker" and guess.reference is not None
    assert abs(guess.flat_w - 52) < 4 and abs(guess.flat_l - 71) < 4
    g = guess.to_garment("t1", "student")
    assert g.outline and len(g.outline) >= 6
    opts = generate_options(g)
    assert opts and all(o.a > 0 for o in opts if o.mode != "hang")
    # the photographed outline folds to a packet like the template's
    best = next(o for o in opts if o.steps)
    st = run(g, best.steps)
    assert st.width < g.flat_w + 2 * (g.sleeve_len or 0) + 1


def test_pipeline_without_sheet_is_approximate():
    ph = render("jeans", "M", seed=22, sheet=False)
    guess = fake("jeans").analyse(ph.image, size="M")
    assert guess.size_source == "approx" and guess.reference is None
    assert guess.flat_l == pytest.approx(104, abs=0.6)  # scaled to the label's chart length
    assert abs(guess.flat_w - 50) < 6  # proportions from the photo


def test_corrected_category_is_respected():
    ph = render("kurti", "M", seed=5)
    guess = fake("tshirt").analyse(ph.image, size="M", category="kurti")
    assert guess.category == "kurti" and guess.top3[0] == ("kurti", 1.0)
    assert categories()["kurti"]["template"] == "dress"
