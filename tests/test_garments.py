import pytest

from fold_and_place import config
from fold_and_place.garments import household_garments, make_garment, persona_garments


def test_drapes_ignore_size_scaling():
    s = make_garment("saree", "a", "S")
    xl = make_garment("saree", "b", "XL")
    assert (s.flat_w, s.flat_l) == (xl.flat_w, xl.flat_l) == (115, 550)
    assert s.size_source == "fabric_standard"


def test_stitched_garments_scale_with_size():
    m = make_garment("tshirt", "a", "M")
    xl = make_garment("tshirt", "b", "XL")
    assert xl.flat_w > m.flat_w and xl.flat_l > m.flat_l
    assert m.size_source == "size_chart"


def test_measured_override_wins():
    g = make_garment("jeans", "j", flat_w=48.0, flat_l=101.0, size_source="marker")
    assert (g.flat_w, g.flat_l, g.size_source) == (48.0, 101.0, "marker")


def test_unknown_category_and_source_rejected():
    with pytest.raises(KeyError):
        make_garment("spacesuit", "x")
    with pytest.raises(ValueError):
        make_garment("tshirt", "x", size_source="guess")


def test_care_flags():
    assert make_garment("lehenga", "l").fold_flat_only
    assert make_garment("lehenga", "l").no_load_on_top
    assert not make_garment("tshirt", "t").fold_flat_only
    assert make_garment("saree", "s", fabric="silk").tissue_layers


def test_every_category_uses_a_known_fabric_and_template():
    fabrics = config.fabrics()
    for name, c in config.categories().items():
        assert c["fabric"] in fabrics, name
        assert c["template"] in {"tee", "dress", "skirt", "trousers", "rect"}, name
    assert len(config.categories()) == 31


def test_personas_are_reproducible_and_owned():
    a = household_garments(seed=3)
    b = household_garments(seed=3)
    assert [(g.id, g.value) for g in a] == [(g.id, g.value) for g in b]
    assert {g.owner for g in a} == set(config.personas())
    assert len({g.id for g in a}) == len(a)


def test_persona_scale():
    base = persona_garments("student")
    double = persona_garments("student", scale=2.0)
    assert len(double) == 2 * len(base)
