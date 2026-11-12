"""Garment silhouettes on a top-down grid: turning the garment upright, measuring it, and
extracting its outline as a polygon in centimetres (so the fold engine can fold *this* garment).

Real photos are taken at any angle and the reference sheet can lie at any angle, so after the
photo is mapped to a top-down grid the garment can be rotated arbitrarily. We find the rotation
in two steps:

1. the small angle that squares the garment's minimum-area rectangle with the axes;
2. which of the four quarter turns is "upright", scored by left-right mirror symmetry (garments
   are mirror-symmetric about their vertical axis) plus a cue per template: sleeves at the top
   for tops, two legs at the bottom for trousers, a wider hem for skirts, length vertical for
   rectangles.

Only numpy and shapely are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon

from fold_and_place.vision.measure import MAX_GARMENT_CM, Measured, apply_h, measure_upright


# ---- grids -----------------------------------------------------------------------------------
def rectify(mask: np.ndarray, H: np.ndarray, res: float, pad: float = 2.0) -> np.ndarray:
    """Sample ``mask`` (photo pixels) onto a grid in the target plane (cell = ``res`` units)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((1, 1), bool)
    step = max(1, len(xs) // 40000)
    pts = apply_h(H, np.c_[xs[::step], ys[::step]])
    x0, y0 = pts.min(axis=0) - pad
    x1, y1 = pts.max(axis=0) + pad
    gx = np.arange(x0, x1, res)
    gy = np.arange(y0, y1, res)
    if len(gx) * len(gy) > 9e6:
        raise ValueError("grid too large: is the reference detected correctly?")
    X, Y = np.meshgrid(gx, gy)
    back = apply_h(np.linalg.inv(H), np.c_[X.ravel(), Y.ravel()])
    u = np.round(back[:, 0]).astype(int)
    v = np.round(back[:, 1]).astype(int)
    ok = (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0])
    out = np.zeros(X.size, bool)
    out[ok] = mask[v[ok], u[ok]]
    return out.reshape(X.shape)


def crop(grid: np.ndarray) -> np.ndarray:
    r = np.nonzero(grid.any(axis=1))[0]
    c = np.nonzero(grid.any(axis=0))[0]
    if len(r) == 0:
        return grid
    return grid[r.min() : r.max() + 1, c.min() : c.max() + 1]


def rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---- orientation -----------------------------------------------------------------------------
def _runs(row: np.ndarray) -> int:
    r = row.astype(np.int8)
    return int((np.diff(np.r_[0, r]) == 1).sum())


def mirror_iou(g: np.ndarray) -> float:
    g = crop(g)
    cols = np.nonzero(g)[1]
    if len(cols) == 0:
        return 0.0
    c = cols.mean()  # mirror about the centroid column
    shift = int(round(2 * c - (g.shape[1] - 1)))
    f = g[:, ::-1]
    if shift > 0:
        f = np.pad(f, ((0, 0), (shift, 0)))[:, : g.shape[1]]
    elif shift < 0:
        f = np.pad(f, ((0, 0), (0, -shift)))[:, -shift:]
    inter = (g & f).sum()
    union = (g | f).sum()
    return float(inter / union) if union else 0.0


def upright_cue(g: np.ndarray, template: str) -> float:
    """Positive when the garment's top is at the top of the grid (row 0)."""
    g = crop(g)
    h, w = g.shape
    if h < 4 or w < 4:
        return 0.0
    widths = g.sum(axis=1) / max(w, 1)
    k = max(1, int(0.25 * h))
    top, bot = widths[:k].mean(), widths[-k:].mean()
    if template in ("tee", "dress"):
        # The armpit is a sharp step in width (sleeves end) near the top; a flared hem is a
        # gradual change. Without sleeves: a top is as wide at the hem as at the chest, a dress
        # is wider at the hem.
        lag = max(1, int(0.02 * h))
        kern = np.ones(lag) / lag
        prof = np.convolve(widths, kern, mode="same")
        d = prof[lag:] - prof[:-lag]  # change going down
        n = len(d)
        top_step = -d[: max(1, int(0.45 * n))].min(initial=0.0)
        bot_step = d[int(0.55 * n) :].max(initial=0.0)
        step = 3.0 * (top_step - bot_step)
        shape_cue = (top - bot) if template == "tee" else 0.8 * (bot - top)
        return step + shape_cue + 0.15 * (h >= w)
    if template == "trousers":
        legs_bot = np.mean([_runs(r) >= 2 for r in g[-k:]])
        legs_top = np.mean([_runs(r) >= 2 for r in g[:k]])
        return 1.5 * (legs_bot - legs_top) + 0.3 * (h > w)
    if template == "skirt":
        return 2.0 * (bot - top) + 0.2 * (w <= 2.2 * h)
    return 0.3 if h >= w else -0.3  # rectangles: length runs top to bottom


def orient(mask: np.ndarray, H: np.ndarray, template: str, coarse: float = 1.0) -> np.ndarray:
    """Rotation (3x3, in the target plane) that makes the garment upright, composed as R @ H."""
    ys, xs = np.nonzero(mask)
    step = max(1, len(xs) // 4000)
    pts = apply_h(H, np.c_[xs[::step], ys[::step]])
    rect = MultiPoint(pts).minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return np.eye(3)
    c = np.asarray(rect.exterior.coords)[:2]
    a = np.arctan2(c[1, 1] - c[0, 1], c[1, 0] - c[0, 0])
    base = -((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)  # smallest turn that squares it up
    centre = pts.mean(axis=0)
    T = np.array([[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1]])
    size = np.ptp(pts, axis=0).max()
    res = max(coarse, size / 160)
    best, best_s = None, -np.inf
    for k in range(4):
        R = rot(base + k * np.pi / 2) @ T
        g = rectify(mask, R @ H, res)
        s = mirror_iou(g) + upright_cue(g, template) + (0.05 if k == 0 else 0.0)  # tiny preference: as shot
        if s > best_s:
            best, best_s = R, s
    return best


# ---- outline -----------------------------------------------------------------------------------
def symmetrise(g: np.ndarray) -> np.ndarray:
    """Average the garment with its mirror image (centred on its centroid column): a real garment
    never lies perfectly straight, and lopsided outlines give lopsided folds."""
    g = crop(g)
    cols = np.nonzero(g)[1]
    c = cols.mean()
    half = int(np.ceil(max(c, g.shape[1] - 1 - c))) + 1
    W = 2 * half + 1
    out = np.zeros((g.shape[0], W), np.int8)
    off = half - int(round(c))
    out[:, off : off + g.shape[1]] = g
    s = out + out[:, ::-1]
    sym = s >= 1  # union of the two halves keeps thin parts (sleeves, straps) ...
    core = s >= 2  # ... but only where they are consistent within a couple of cells
    grow = core.copy()
    for d in (1, 2):
        grow[:, d:] |= core[:, :-d]
        grow[:, :-d] |= core[:, d:]
        grow[d:, :] |= core[:-d, :]
        grow[:-d, :] |= core[d:, :]
    return crop(sym & grow)


def grid_outline(g: np.ndarray, res: float, simplify: float = 0.6) -> Polygon:
    """Polygon (cm) of an upright grid: hem at y = 0, centred on x = 0, holes removed."""
    g = crop(g)
    h, w = g.shape
    boxes = []
    for i in range(h):
        row = g[i].astype(np.int8)
        d = np.diff(np.r_[0, row, 0])
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        y0 = (h - 1 - i) * res
        for s, e in zip(starts, ends):
            boxes.append(shapely.box(s * res, y0, e * res, y0 + res))
    if not boxes:
        return Polygon()
    shape = shapely.union_all(boxes).buffer(res * 0.6).buffer(-res * 0.6)
    parts = list(shapely.get_parts(shape))
    poly = max(parts, key=lambda p: p.area)
    poly = Polygon(poly.exterior).simplify(simplify, preserve_topology=True)
    x0, y0, x1, _ = poly.bounds
    return shapely.affinity.translate(poly, -(x0 + x1) / 2, -y0)


# ---- one call ----------------------------------------------------------------------------------
@dataclass
class Silhouette:
    measured: Measured
    outline: list[tuple[float, float]]  # cm, hem at y=0, centred
    grid: np.ndarray = field(repr=False)  # upright top-down mask, ``res`` cm per cell
    res: float = 0.5
    scale_known: bool = True


def silhouette(mask: np.ndarray, template: str, H: np.ndarray | None = None, *, length_cm: float | None = None,
               res: float = 0.5) -> Silhouette:
    """Upright, measured silhouette.

    ``H`` maps photo pixels to cm on the table (from a reference sheet). Without it the photo is
    measured in pixels and scaled so that its length equals ``length_cm`` (the size label's chart
    length): the proportions are real, the scale is approximate.
    """
    import shapely.affinity  # noqa: F401  (submodule used in grid_outline)

    known = H is not None
    if H is None:
        ys, xs = np.nonzero(mask)
        span = max(np.ptp(xs), np.ptp(ys), 1)
        H = np.diag([1.0, 1.0, 1.0])
        res_px = span / 300
    R = orient(mask, H, template)
    HR = R @ H
    if known:
        g = crop(rectify(mask, HR, res))
        cell = res
    else:
        g = crop(rectify(mask, HR, res_px))
        cell = res_px
    m = measure_upright(g, template, cell)
    if not known:
        f = (length_cm or m.flat_l) / max(m.flat_l, 1e-9)
        m = Measured(m.flat_w * f, m.flat_l * f, m.sleeve_len * f, None if m.hem_w is None else m.hem_w * f)
        cell *= f
    if max(m.flat_w, m.flat_l) > MAX_GARMENT_CM:
        raise ValueError("measured an impossible size: is the reference detected correctly?")
    poly = grid_outline(symmetrise(g), cell, simplify=max(0.4, 1.2 * cell))
    pts = [(round(x, 2), round(y, 2)) for x, y in np.asarray(poly.exterior.coords)[:-1]]
    return Silhouette(m, pts, g, cell, known)
