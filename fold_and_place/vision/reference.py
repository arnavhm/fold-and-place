"""Find the reference object in a real photo, automatically.

* ``detect_a4`` - any sheet of A4 paper lying flat next to the garment. We look for bright,
  low-saturation four-sided regions (and, for white paper on a pale sheet, for four-sided edge
  loops), keep convex quads whose sides have roughly the A4 ratio (1:1.414, allowing for
  perspective) and that are mostly filled, and score them by size and whiteness. Corners are
  refined to sub-pixel accuracy.
* ``detect_aruco`` - a printed ArUco marker (DICT_4X4_50) of known size.
* ``find_reference`` - ArUco first (exact), then A4.

Each returns ``(corners_px 4x2, corners_cm 4x2)`` in matching order, or None. Needs OpenCV.
"""

from __future__ import annotations

import numpy as np

A4 = (21.0, 29.7)


def _order(q: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left (image coordinates)."""
    q = np.asarray(q, float).reshape(4, 2)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]  # clockwise on screen, starting near -pi (left)
    start = int(np.argmin(q.sum(axis=1)))
    return np.roll(q, -start, axis=0)


def _cm_for(q: np.ndarray) -> np.ndarray:
    top = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    side = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    w, h = (A4[1], A4[0]) if top > side else (A4[0], A4[1])
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], float)


def _to_bgr(img) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if hasattr(img, "mode"):  # PIL image: RGB
        a = a[:, :, :3][:, :, ::-1]
    return np.ascontiguousarray(a[:, :, :3])


def detect_a4(img, min_frac: float = 0.001, debug: bool = False):
    """``img``: PIL image (RGB) or BGR array."""
    import cv2

    bgr = _to_bgr(img)
    H0, W0 = bgr.shape[:2]
    scale = 1000.0 / max(H0, W0)
    small = cv2.resize(bgr, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()
    scale = min(scale, 1.0)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    area_img = small.shape[0] * small.shape[1]

    maps = []
    blur = cv2.GaussianBlur(v, (5, 5), 0)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for t in sorted({int(otsu), int(min(otsu + 25, 235)), 200, 170}):
        m = ((blur >= t) & (s <= 70)).astype(np.uint8) * 255
        maps.append(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    maps.append(edges)

    cands = []
    for k, m in enumerate(maps):
        mode = cv2.RETR_LIST if k == len(maps) - 1 else cv2.RETR_EXTERNAL
        contours, _ = cv2.findContours(m, mode, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            a = cv2.contourArea(c)
            if a < min_frac * area_img or a > 0.6 * area_img:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            q = None
            for eps in (0.015, 0.025, 0.04):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4:
                    q = ap.reshape(4, 2).astype(float)
                    break
            if q is None or not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
                continue
            q = _order(q)
            sides = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
            a_side, b_side = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
            ratio = max(a_side, b_side) / max(min(a_side, b_side), 1e-6)
            opp = max(sides[0], sides[2]) / max(min(sides[0], sides[2]), 1e-6), max(sides[1], sides[3]) / max(min(sides[1], sides[3]), 1e-6)
            if not (1.15 <= ratio <= 1.8) or max(opp) > 1.4:
                continue
            ang = []
            for i in range(4):
                u, w_ = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
                ang.append(np.degrees(np.arccos(np.clip(u @ w_ / (np.linalg.norm(u) * np.linalg.norm(w_) + 1e-9), -1, 1))))
            if min(ang) < 55 or max(ang) > 125:
                continue
            qa = cv2.contourArea(q.astype(np.float32))
            poly_mask = np.zeros(v.shape, np.uint8)
            cv2.fillConvexPoly(poly_mask, q.astype(np.int32), 1)
            inner = poly_mask.astype(bool)
            fill = a / max(qa, 1e-6) if k < len(maps) - 1 else 1.0
            if fill < 0.8:
                continue
            bright = float(v[inner].mean()) / 255
            sat = float(s[inner].mean()) / 255
            uniform = 1.0 - min(1.0, float(v[inner].std()) / 60)
            if bright < 0.45 or sat > 0.22:
                continue
            white = max(0.0, bright * (1 - 2.5 * sat))
            score = (qa / area_img) ** 0.25 * white**3 * uniform * (1.0 - abs(ratio - 1.414) / 1.414)
            cands.append((score, q))
    if not cands:
        return None
    cands.sort(key=lambda t: -t[0])
    q = cands[0][1] / scale
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    win = max(3, int(4 / scale))
    try:
        q = cv2.cornerSubPix(gray_full, q.astype(np.float32).reshape(-1, 1, 2), (win, win), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)).reshape(4, 2)
    except cv2.error:  # pragma: no cover
        pass
    q = _order(q)
    out = (q.astype(float), _cm_for(q))
    return (out, cands) if debug else out


def detect_aruco(img, marker_cm: float = 10.0):
    import cv2

    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(_to_bgr(img))
    if ids is None or len(corners) == 0:
        return None
    s = marker_cm
    return corners[0].reshape(4, 2).astype(float), np.array([[0, 0], [s, 0], [s, s], [0, s]], float)


def find_reference(img):
    """(corners_px, corners_cm, kind) or None. Never raises: no OpenCV means no reference."""
    try:
        r = detect_aruco(img)
        if r is not None:
            return r[0], r[1], "aruco"
        r = detect_a4(img)
        if r is not None:
            return r[0], r[1], "a4"
    except ImportError:
        return None
    return None


def main_part(mask: np.ndarray, min_frac: float = 0.15) -> np.ndarray:
    """Keep the garment only: components at least ``min_frac`` of the largest (e.g. after the
    reference sheet was cut out, its leftover border slivers go)."""
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover
        return mask
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.nonzero(sizes >= min_frac * sizes.max())[0])


def clean_mask(mask: np.ndarray, min_frac: float = 0.02) -> np.ndarray:
    """Keep the garment: drop specks, fill holes (prints, logos), keep the large components."""
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover
        return mask
    m = ndimage.binary_opening(mask, iterations=2)
    m = ndimage.binary_closing(m, iterations=3)
    lab, n = ndimage.label(m)
    if n == 0:
        return mask
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes >= max(min_frac * sizes.max(), 1))[0])
    return ndimage.binary_fill_holes(keep)
