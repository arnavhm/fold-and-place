"""Realistic synthetic photos for testing the real-photo path without anyone's clothes.

A garment (our outline for its category, in cm, with edge wobble for wrinkles and a print) and
a sheet of A4 paper lie on a textured bedsheet. A phone camera looks down from a random height
with a random tilt (up to ``max_tilt``) and a random turn (any angle), under uneven light. We add
soft shadows, sensor noise, blur and JPEG compression. The true sizes are known exactly.
Needs OpenCV and Pillow.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from fold_and_place.folding.templates import outline
from fold_and_place.garments.record import Garment, make_garment
from fold_and_place.vision.measure import apply_h

A4 = (21.0, 29.7)


@dataclass
class SynthPhoto:
    image: object  # PIL.Image (RGB)
    garment: Garment
    sheet_px: np.ndarray | None  # true corners in the photo (None when no sheet)
    H_cm_to_px: np.ndarray


def camera(tilt_deg: float, yaw_deg: float, dist_cm: float, f_px: float, size: tuple[int, int]) -> np.ndarray:
    t, y = np.radians(tilt_deg), np.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R = Rx @ Rz
    K = np.array([[f_px, 0, size[0] / 2], [0, f_px, size[1] / 2], [0, 0, 1]])
    H = K @ np.c_[R[:, 0], R[:, 1], [0.0, 0.0, dist_cm]]
    return H / H[2, 2]


def _wobble(poly: np.ndarray, rng, amp: float) -> np.ndarray:
    """Densify the outline and push it in and out a little (wrinkles, imperfect laying)."""
    pts = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(2, int(np.linalg.norm(b - a) / 2.0))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.append(a + t * (b - a))
    p = np.concatenate(pts)
    c = p.mean(axis=0)
    r = p - c
    k = np.arange(len(p))
    noise = sum(rng.uniform(-1, 1) * np.sin(2 * np.pi * f * k / len(p) + rng.uniform(0, 6.28)) for f in (3, 7, 13)) / 3
    return p + r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9) * (amp * noise)[:, None]


def render(category: str, size: str = "M", *, seed: int = 0, sheet: bool = True, max_tilt: float = 25.0,
           img_size: tuple[int, int] = (1600, 1200), dark_sheet: bool | None = None) -> SynthPhoto:
    import cv2
    from PIL import Image

    rng = np.random.default_rng(seed)
    g = make_garment(category, f"synth_{category}", size)
    poly = np.asarray(outline(g).exterior.coords)[:-1]
    poly = np.c_[poly[:, 0], -(poly[:, 1] - g.flat_l / 2)]  # cm, y down, centred
    poly = _wobble(poly, rng, amp=0.012 * max(g.flat_w, g.flat_l))
    # the sheet lies next to the garment, at any angle
    side = rng.integers(4)
    a = rng.uniform(0, 2 * np.pi)
    S = np.array([[0, 0], [A4[0], 0], [A4[0], A4[1]], [0, A4[1]]]) - np.array(A4) / 2
    S = S @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
    gap = 8 + 0.5 * np.ptp(S, axis=0).max()
    off = {0: [poly[:, 0].max() + gap, 0], 1: [poly[:, 0].min() - gap, 0],
           2: [0, poly[:, 1].max() + gap], 3: [0, poly[:, 1].min() - gap]}[int(side)]
    S = S + np.array(off) + rng.uniform(-5, 5, 2)
    scene = np.r_[poly, S] if sheet else poly
    centre = (scene.min(axis=0) + scene.max(axis=0)) / 2
    poly, S = poly - centre, S - centre
    extent = np.ptp(np.r_[poly, S] if sheet else poly, axis=0)
    dist = 150.0
    yaw = rng.uniform(0, 360)
    # frame the scene in ~70-85% of the photo whatever the turn
    diag = np.hypot(*extent)
    f = rng.uniform(0.6, 0.75) * min(img_size) / diag * dist
    H = camera(rng.uniform(0, max_tilt), yaw, dist, f, img_size)

    W, Hh = img_size
    dark = bool(rng.random() < 0.4) if dark_sheet is None else dark_sheet
    base = rng.uniform(40, 80, 3) if dark else rng.uniform(150, 205, 3)
    img = np.ones((Hh, W, 3)) * base
    yy, xx = np.mgrid[0:Hh, 0:W]
    img += (np.sin(xx / rng.uniform(3, 6)) * np.sin(yy / rng.uniform(3, 6)))[..., None] * rng.uniform(2, 6)  # weave
    blot = cv2.GaussianBlur(rng.normal(0, 1, (Hh // 16, W // 16)), (0, 0), 3)
    img += cv2.resize(blot, (W, Hh))[..., None] * 10
    # garment colour contrasting with the sheet, with a print
    gcol = rng.uniform(150, 235, 3) if dark else rng.uniform(25, 120, 3)
    if rng.random() < 0.3:
        gcol = rng.uniform(0, 255, 3) * np.array([1, 0.3, 0.3])[rng.permutation(3)] + (0 if dark else 0)
        gcol = np.clip(gcol + (120 if dark else 0), 0, 255)
    gp = apply_h(H, poly).astype(np.int32)
    sp = apply_h(H, S) if sheet else None
    shadow = np.zeros((Hh, W), np.float32)
    cv2.fillPoly(shadow, [gp + np.array([6, 8])], 1.0)
    if sheet:
        cv2.fillPoly(shadow, [sp.astype(np.int32) + np.array([4, 6])], 0.8)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 9)
    img *= (1 - 0.25 * shadow)[..., None]
    gm = np.zeros((Hh, W), np.uint8)
    cv2.fillPoly(gm, [gp], 1)
    gimg = np.ones_like(img) * gcol
    if rng.random() < 0.5:  # stripes or dots print
        per = rng.uniform(15, 40)
        pat = (np.sin((xx * np.cos(a) + yy * np.sin(a)) / per * 2 * np.pi) > 0.6)[..., None]
        gimg = np.where(pat, gimg * 0.75 + 30, gimg)
    folds = cv2.GaussianBlur(rng.normal(0, 1, (Hh // 24, W // 24)), (0, 0), 2)
    gimg += cv2.resize(folds, (W, Hh))[..., None] * 14  # creases
    img = np.where(gm[..., None] > 0, gimg, img)
    if sheet:
        smask = np.zeros((Hh, W), np.uint8)
        cv2.fillPoly(smask, [np.round(sp).astype(np.int32)], 1)
        paper = np.ones_like(img) * rng.uniform(232, 250) * np.array([1.0, 1.0, rng.uniform(0.95, 1.0)])
        img = np.where(smask[..., None] > 0, paper, img)
    # light falls off across the photo
    cx, cy = rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * Hh
    light = 1.0 - 0.25 * (((xx - cx) / W) ** 2 + ((yy - cy) / Hh) ** 2)
    img = img * light[..., None] * rng.uniform(0.85, 1.1)
    img += rng.normal(0, 3.0, img.shape)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (0, 0), rng.uniform(0.6, 1.3))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(rng.integers(70, 92)))
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return SynthPhoto(im, g, sp, H)
