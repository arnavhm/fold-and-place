"""From a garment mask to centimetres.

Priority (the first that applies wins, and is recorded on the garment):
  1. marker      - a printed square marker or an A4 sheet lying next to the garment: a homography
                   maps the photo to a top-down cm grid. Valid for flat-laid garments only (they
                   lie in the marker's plane).
  2. size_chart  - stitched garments: one length from the size chart for the chosen size, the
                   width from the mask's proportions (flat-lay) or the chart too (worn photos).
  3. fabric_standard - drapes (saree, dupatta, dhoti, lungi): standard fabric size; a draped saree
                   never shows its 5.5 m length in a photo.
Only numpy and shapely are needed here; marker DETECTION needs OpenCV (optional).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fold_and_place import config
from fold_and_place.garments.record import make_garment

A4_CM = (21.0, 29.7)
MAX_GARMENT_CM = 800.0  # nothing in a wardrobe is wider than 8 m laid flat


# ---- homography -------------------------------------------------------------------------
def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 H with dst ~ H @ src (direct linear transform, >= 4 point pairs, normalised)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)

    def normalise(p):
        c = p.mean(axis=0)
        s = np.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        ph = np.c_[p, np.ones(len(p))] @ T.T
        return ph[:, :2], T

    s_n, Ts = normalise(src)
    d_n, Td = normalise(dst)
    rows = []
    for (x, y), (u, v) in zip(s_n, d_n):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ph = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


def rectify_mask(mask: np.ndarray, H_px_to_cm: np.ndarray, res_cm: float = 0.25, pad_cm: float = 2.0) -> np.ndarray:
    """Top-down mask on a cm grid (one cell = ``res_cm``), sampled through the inverse homography."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((1, 1), bool)
    cm = apply_h(H_px_to_cm, np.c_[xs, ys])
    if np.ptp(cm[:, 0]) > MAX_GARMENT_CM or np.ptp(cm[:, 1]) > MAX_GARMENT_CM:
        raise ValueError("homography maps the garment to an impossible size: is the marker detected correctly?")
    x0, y0 = cm.min(axis=0) - pad_cm
    x1, y1 = cm.max(axis=0) + pad_cm
    gx = np.arange(x0, x1, res_cm)
    gy = np.arange(y0, y1, res_cm)
    X, Y = np.meshgrid(gx, gy)
    back = apply_h(np.linalg.inv(H_px_to_cm), np.c_[X.ravel(), Y.ravel()])
    u = np.round(back[:, 0]).astype(int)
    v = np.round(back[:, 1]).astype(int)
    ok = (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0])
    out = np.zeros(X.size, bool)
    out[ok] = mask[v[ok], u[ok]]
    return out.reshape(X.shape)


# ---- shape measurements on an upright cm-grid mask ---------------------------------------
@dataclass
class Measured:
    flat_w: float
    flat_l: float
    sleeve_len: float = 0.0
    hem_w: float | None = None


def measure_upright(grid: np.ndarray, template: str, res_cm: float = 0.25) -> Measured:
    """Measure a top-down mask whose hem is at the bottom (row index grows downwards)."""
    rows = np.nonzero(grid.any(axis=1))[0]
    if len(rows) == 0:
        return Measured(0.0, 0.0)
    top, bot = rows.min(), rows.max()
    length = (bot - top + 1) * res_cm
    widths = grid.sum(axis=1) * res_cm
    band = lambda a, b: widths[int(top + a * (bot - top)) : int(top + b * (bot - top)) + 1]  # noqa: E731
    total_w = (np.ptp(np.nonzero(grid.any(axis=0))[0]) + 1) * res_cm
    if template in ("tee", "dress"):
        body = float(np.median(band(0.6, 0.95)))
        if template == "dress":
            # chest: just below the armpit (where the sleeves end), as a tailor measures it
            prof = widths[top : bot + 1]
            n = len(prof)
            lag = max(1, int(0.02 * n))
            upper = prof[: max(lag + 2, int(0.6 * n))]
            drops = upper[:-lag] - upper[lag:]
            if len(drops) and drops.max() > 0.08 * prof.max():
                a = int(np.argmax(drops)) + lag
                lo, hi = a + max(1, int(0.01 * n)), a + max(2, int(0.06 * n))
                body = float(np.median(prof[lo:hi])) if hi > lo else float(prof[a])
            else:  # no clear armpit step: the narrowest point of the upper body
                body = float(np.min(prof[int(0.15 * n) : max(int(0.15 * n) + 1, int(0.45 * n))]))
            hem = float(np.median(band(0.97, 1.0)))
        sleeve = max(0.0, (total_w - body) / 2) if total_w > body * 1.15 else 0.0
        return Measured(body, length, sleeve, hem if template == "dress" else None)
    if template == "trousers":
        return Measured(float(np.max(band(0.0, 0.15))), length)
    if template == "skirt":
        return Measured(float(np.percentile(band(0.0, 0.06), 90)), length, hem_w=float(np.median(band(0.97, 1.0))))
    return Measured(min(total_w, length), max(total_w, length))


def drop_marker(mask: np.ndarray, marker_px: np.ndarray, grow_px: float = 6.0) -> np.ndarray:
    """Remove the reference sheet / marker (and a small border) from a garment mask."""
    from matplotlib.path import Path as MplPath

    c = np.asarray(marker_px, float)
    centre = c.mean(axis=0)
    c = centre + (c - centre) * (1 + grow_px / max(np.linalg.norm(c - centre, axis=1).mean(), 1.0))
    ys, xs = np.nonzero(mask)
    inside = MplPath(c).contains_points(np.c_[xs, ys])
    out = mask.copy()
    out[ys[inside], xs[inside]] = False
    return out


def measure_with_marker(
    mask: np.ndarray, marker_px: np.ndarray, marker_cm: np.ndarray, template: str, res_cm: float = 0.25
) -> Measured:
    """``marker_px``: the 4 detected corners in the photo; ``marker_cm``: the same corners on the
    table in cm (e.g. a 10 cm square, or an A4 sheet)."""
    H = homography(marker_px, marker_cm)
    return measure_upright(rectify_mask(drop_marker(mask, marker_px), H, res_cm), template, res_cm)


def detect_marker(image_bgr: np.ndarray, marker_cm: float = 10.0):  # pragma: no cover - needs OpenCV
    """Find an ArUco marker (DICT_4X4_50) with OpenCV. Returns (corners_px 4x2, corners_cm 4x2) or None."""
    import cv2

    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(image_bgr)
    if ids is None or len(corners) == 0:
        return None
    px = corners[0].reshape(4, 2)
    s = marker_cm
    return px, np.array([[0, 0], [s, 0], [s, s], [0, s]], float)


# ---- fallbacks without a marker ------------------------------------------------------------
def size_from_chart(category: str, size: str, mask: np.ndarray | None, photo_type: str) -> tuple[float, float, str]:
    """(flat_w, flat_l, source). Drapes -> fabric standard; stitched -> chart length, and width from
    the mask's aspect ratio on flat-lay photos (worn photos keep the chart width)."""
    g = make_garment(category, "probe", size)
    if g.is_drape:
        return g.flat_w, g.flat_l, "fabric_standard"
    if mask is None or photo_type != "flat" or g.template not in ("rect", "skirt", "trousers"):
        return g.flat_w, g.flat_l, "size_chart"
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(rows) < 5 or len(cols) < 5:
        return g.flat_w, g.flat_l, "size_chart"
    aspect = (np.ptp(cols) + 1) / (np.ptp(rows) + 1)
    return round(g.flat_l * aspect, 1), g.flat_l, "size_chart"


def default_size() -> str:
    return "M" if "M" in config.size_scale() else next(iter(config.size_scale()))
