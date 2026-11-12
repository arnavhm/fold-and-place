"""Photo helpers for the Clothes page: sample photos, cached analysis, the "what the app saw"
overlay, the measured outline with dimensions, and thumbnails."""

from __future__ import annotations

import base64
import hashlib
import io

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from fold_and_place.app.vision_status import vision_status

SAMPLE_SPECS = [  # (category, seed, sheet?) - realistic synthetic photos with known true sizes
    ("tshirt", 10, True), ("kurti", 35, True), ("jeans", 12, True), ("skirt", 24, True),
    ("shirt", 31, True), ("palazzo", 29, True), ("trousers", 4, False),
]


@st.cache_data(show_spinner=False)
def sample_photos() -> list[dict]:
    """Rendered on this machine (no downloads): each knows its true size for comparison."""
    out = []
    try:
        from fold_and_place.vision.synth import render
    except ImportError:  # pragma: no cover
        return out
    for cat, seed, sheet in SAMPLE_SPECS:
        try:
            ph = render(cat, "M", seed=seed, sheet=sheet)
        except ImportError:
            return []
        buf = io.BytesIO()
        ph.image.save(buf, "JPEG", quality=90)
        out.append({"name": f"{ph.garment.label}{'' if sheet else ' (no A4 sheet)'}", "bytes": buf.getvalue(),
                    "true": (ph.garment.flat_w, ph.garment.flat_l), "category": cat})
    return out


@st.cache_resource(show_spinner="Loading the fashion model (the first time takes a while)...")
def analyzer():
    from fold_and_place.vision import Analyzer

    return Analyzer()


def analyse(data: bytes, size: str, category: str | None):
    """Full analysis when the recognition models are installed, else measuring only (needs the
    category). Cached per photo + settings."""
    key = (hashlib.md5(data).hexdigest(), size, category)
    cache = st.session_state.setdefault("_analysis", {})
    if key in cache:
        return cache[key]
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1800, 1800))
    if vision_status()["recognise"]:
        guess = analyzer().analyse(img, size=size, category=category)
    else:
        from fold_and_place.vision.pipeline import measure_photo

        guess = measure_photo(img, category or "tshirt", size)
    cache[key] = (guess, img)
    return guess, img


def overlay(img: Image.Image, guess) -> Image.Image:
    """The photo with the found A4 sheet (green) and the garment's edge (orange)."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im, "RGBA")
    if guess.mask is not None:
        try:
            import cv2

            cs, _ = cv2.findContours(guess.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if len(c) > 10:
                    pts = [tuple(map(int, p[0])) for p in c]
                    d.line(pts + [pts[0]], fill=(232, 120, 40, 255), width=max(3, im.width // 300))
        except ImportError:
            pass
    if guess.reference is not None:
        q = [tuple(map(float, p)) for p in guess.reference[0]]
        d.polygon(q, fill=(60, 170, 90, 70), outline=(40, 150, 70, 255), width=max(3, im.width // 300))
    return im


def outline_svg(guess, width: int = 280) -> str:
    """The upright outline with its measured width and length drawn on."""
    if not guess.outline:
        return ""
    pts = np.asarray(guess.outline, float)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    W, L = x1 - x0, y1 - y0
    pad = 34
    s = (width - 2 * pad) / max(W, L)
    h = L * s + 2 * pad
    P = " ".join(f"{pad + (x - x0) * s + (max(W, L) - W) * s / 2:.1f},{pad + (y1 - y) * s:.1f}" for x, y in pts)
    cx = pad + (max(W, L) - W) * s / 2
    body = guess.flat_w * s
    bx0 = cx + (W * s - body) / 2
    by = pad + L * s * (0.25 if guess.outline and _is_dress(guess) else 0.82)
    approx = "≈ " if guess.size_source != "marker" else ""
    return (
        f'<svg viewBox="0 0 {width} {h:.0f}" width="100%" style="max-width:{width}px" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">'
        f'<polygon points="{P}" fill="#F3E3D3" stroke="#B4532A" stroke-width="2"/>'
        f'<line x1="{bx0:.1f}" y1="{by:.1f}" x2="{bx0 + body:.1f}" y2="{by:.1f}" stroke="#2B2420" stroke-width="1.5" marker-start="url(#a)" marker-end="url(#a)"/>'
        f'<text x="{bx0 + body / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" font-size="13" font-weight="600" fill="#2B2420">{approx}{guess.flat_w:.0f} cm</text>'
        f'<line x1="{width - pad + 14:.1f}" y1="{pad:.1f}" x2="{width - pad + 14:.1f}" y2="{pad + L * s:.1f}" stroke="#2B2420" stroke-width="1.5" marker-start="url(#a)" marker-end="url(#a)"/>'
        f'<text x="{width - pad + 8:.1f}" y="{pad + L * s / 2:.1f}" text-anchor="end" font-size="13" font-weight="600" fill="#2B2420" transform="rotate(-90 {width - pad + 8:.1f} {pad + L * s / 2:.1f})">{approx}{guess.flat_l:.0f} cm</text>'
        '<defs><marker id="a" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#2B2420"/></marker></defs>'
        "</svg>"
    )


def _is_dress(guess) -> bool:
    from fold_and_place import config

    return config.categories()[guess.category]["template"] == "dress"


def thumb_uri(img: Image.Image, size: int = 160) -> str:
    im = img.convert("RGB").copy()
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
