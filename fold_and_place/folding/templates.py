"""Flat-lay outline polygons for each garment template.

Frame: x runs across the garment (centred on x = 0), y runs from the hem (y = 0) up to the
neckline / waistband (y = flat_l). Units: cm.
"""

from __future__ import annotations

from shapely.geometry import Polygon

from fold_and_place.garments.record import Garment


def tee(g: Garment) -> Polygon:
    """T-shape: body rectangle with sleeves laid straight out at the shoulders."""
    a, L = g.flat_w / 2, g.flat_l
    s, sw = g.sleeve_len, min(g.sleeve_w, g.flat_l * 0.6)
    if s <= 0 or sw <= 0:
        return Polygon([(-a, 0), (a, 0), (a, L), (-a, L)])
    return Polygon(
        [(-a, 0), (a, 0), (a, L - sw), (a + s, L - sw), (a + s, L), (-a - s, L), (-a - s, L - sw), (-a, L - sw)]
    )


def dress(g: Garment) -> Polygon:
    """Like ``tee`` but the body flares from the armpit down to ``hem_w``."""
    a, L = g.flat_w / 2, g.flat_l
    h = (g.hem_w or g.flat_w) / 2
    s, sw = g.sleeve_len, min(g.sleeve_w, g.flat_l * 0.4)
    if s <= 0 or sw <= 0:
        return Polygon([(-h, 0), (h, 0), (a, L), (-a, L)])
    return Polygon(
        [(-h, 0), (h, 0), (a, L - sw), (a + s, L - sw), (a + s, L), (-a - s, L), (-a - s, L - sw), (-a, L - sw)]
    )


def skirt(g: Garment) -> Polygon:
    """Trapezoid: waist ``flat_w`` at the top, ``hem_w`` at the bottom."""
    a, L = g.flat_w / 2, g.flat_l
    h = (g.hem_w or g.flat_w) / 2
    return Polygon([(-h, 0), (h, 0), (a, L), (-a, L)])


def trousers(g: Garment) -> Polygon:
    """Two legs joined at the seat; each leg hem is 40% of the hip width."""
    a, L = g.flat_w / 2, g.flat_l
    rise = g.rise or 0.27 * L
    gap = max(1.0, a - 0.4 * g.flat_w)
    return Polygon([(-a, 0), (-gap, 0), (0, L - rise), (gap, 0), (a, 0), (a, L), (-a, L)])


def rect(g: Garment) -> Polygon:
    a, L = g.flat_w / 2, g.flat_l
    return Polygon([(-a, 0), (a, 0), (a, L), (-a, L)])


TEMPLATES = {"tee": tee, "dress": dress, "skirt": skirt, "trousers": trousers, "rect": rect}


def outline(g: Garment) -> Polygon:
    try:
        return TEMPLATES[g.template](g)
    except KeyError as exc:
        raise KeyError(f"no outline template {g.template!r}") from exc
