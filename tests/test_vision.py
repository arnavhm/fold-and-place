"""Vision tests run with fake models: no torch, no downloads (the real models are exercised by
scripts/smoke_vision.py on a laptop)."""

import json

import numpy as np
import pytest
from PIL import Image

from fold_and_place.config import ROOT
from fold_and_place.experiments.e3_measure import synth
from fold_and_place.vision import Analyzer
from fold_and_place.vision import labels as L
from fold_and_place.vision.measure import apply_h, homography, measure_with_marker, size_from_chart
from fold_and_place.vision.models import LinearHead, PlainSegmenter, ensemble, zero_shot


class FakeEmbedder:
    """Every label of every prompt bank gets its own axis; the image points at the chosen ones."""

    def __init__(self, picks: dict[str, str]):
        self.picks = picks
        banks = {
            "photo": L.PHOTO_TYPE_PROMPTS,
            "category": L.category_prompts(),
            "fabric": L.FABRIC_PROMPTS,
            "embellish": L.EMBELLISH_PROMPTS,
        }
        self.axis = {}
        for name, prompts in banks.items():
            for lab in prompts:
                self.axis[(name, lab)] = len(self.axis)
        self.banks = banks
        self.dim = len(self.axis)

    def images(self, imgs):
        v = np.zeros((len(imgs), self.dim))
        for name, lab in self.picks.items():
            v[:, self.axis[(name, lab)]] = 1.0
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def prompt_bank(self, prompts):
        name = next(n for n, p in self.banks.items() if set(p) == set(prompts))
        labels = list(prompts)
        bank = np.zeros((len(labels), self.dim))
        for i, lab in enumerate(labels):
            bank[i, self.axis[(name, lab)]] = 1.0
        return labels, bank


def fake(category, photo="flat", fabric="denim", emb="none"):
    return Analyzer(
        embedder=FakeEmbedder({"photo": photo, "category": category, "fabric": fabric, "embellish": emb}),
        flat_seg=PlainSegmenter(45),
        worn_seg=PlainSegmenter(45),
        use_head=False,
    )


def test_homography_recovers_a_known_mapping():
    rng = np.random.default_rng(0)
    H = np.array([[1.2, 0.1, 30], [-0.05, 0.9, 12], [1e-4, 2e-4, 1]])
    src = rng.uniform(0, 500, (8, 2))
    est = homography(src, apply_h(H, src))
    probe = rng.uniform(0, 500, (20, 2))
    assert np.allclose(apply_h(est, probe), apply_h(H, probe), atol=1e-6)


@pytest.mark.parametrize("cat,size", [("tshirt", "M"), ("jeans", "M"), ("skirt", "M")])
def test_marker_measurement_on_synthetic_photo(cat, size):
    rng = np.random.default_rng(1)
    g, mask, mpx, mcm = synth(cat, size, rng, noise_px=0.0)
    m = measure_with_marker(mask, mpx, mcm, g.template)
    assert m.flat_l == pytest.approx(g.flat_l, abs=2.0)
    assert m.flat_w == pytest.approx(g.flat_w, abs=2.0)


def test_label_maps():
    assert L.map_indofashion("women_kurta") == "kurti"
    assert L.map_indofashion("Sherwanis") == "sherwani"
    assert L.map_indofashion("women_mojari") is None  # footwear is skipped
    assert L.map_myntra("Kurtas", "Men") == "kurta_men"
    assert L.map_myntra("Kurtas", "Women") == "kurti"
    assert L.map_myntra("Tshirts") == "tshirt"
    assert set(L.category_prompts()) == set(L.category_prompts())  # deterministic
    assert len(L.category_prompts()) == 31


def test_drapes_use_fabric_standard_and_stitched_use_chart():
    assert size_from_chart("saree", "M", None, "worn")[2] == "fabric_standard"
    w, length, src = size_from_chart("jeans", "L", None, "worn")
    assert src == "size_chart" and length > 100


def test_linear_head_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (40, 6)) + [3, 0, 0, 0, 0, 0], rng.normal(0, 1, (40, 6)) - [3, 0, 0, 0, 0, 0]]
    y = ["saree"] * 40 + ["jeans"] * 40
    head = LinearHead.train(X, y)
    head.save(tmp_path / "h.npz")
    back = LinearHead.load(tmp_path / "h.npz")
    assert np.allclose(back.proba(X), head.proba(X))
    assert (np.array(back.classes)[back.proba(X).argmax(1)] == np.array(y)).mean() > 0.95


def test_ensemble_keeps_zero_shot_for_unseen_classes():
    labels = ["a", "b", "c"]
    zs = zero_shot(np.array([[1.0, 0, 0]]), np.eye(3))
    head = LinearHead(["a", "b"], np.zeros((2, 3)), np.zeros(2))
    p = ensemble(zs, labels, head, head.proba(np.array([[1.0, 0, 0]])))
    assert p.sum() == pytest.approx(1.0)
    assert p[0, 2] == pytest.approx(0.5 * zs[0, 2])


def test_pipeline_with_marker_on_bundled_sample():
    meta = json.loads((ROOT / "assets" / "samples" / "samples.json").read_text())
    m = next(x for x in meta if x["category"] == "jeans")
    img = Image.open(ROOT / "assets" / "samples" / m["file"])
    guess = fake("jeans").analyse(img, size="M", marker=(np.array(m["a4_corners_px"]), np.array(m["a4_corners_cm"])))
    assert guess.category == "jeans" and guess.size_source == "marker"
    assert guess.flat_l == pytest.approx(m["true_flat_l_cm"], abs=2.0)
    g = guess.to_garment("j1", owner="student", value=7)
    assert g.size_source == "marker" and g.fabric == "denim" and g.owner == "student"


def test_pipeline_worn_saree_uses_fabric_standard():
    img = Image.new("RGB", (200, 300), (240, 240, 240))
    img.paste((150, 20, 40), (60, 40, 140, 280))
    guess = fake("saree", photo="worn", fabric="silk_heavy").analyse(img)
    assert guess.size_source == "fabric_standard"
    assert (guess.flat_w, guess.flat_l) == (115, 550)


def test_heavy_categories_default_to_heavy_embellishment():
    img = Image.new("RGB", (200, 300), (240, 240, 240))
    img.paste((180, 30, 60), (40, 40, 160, 280))
    guess = fake("lehenga", photo="flat", fabric="suiting", emb="none").analyse(img)
    assert guess.embellish == "heavy"
    assert guess.to_garment("l").fold_flat_only
