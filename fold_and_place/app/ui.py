"""Look and feel: theme CSS, the step header, garment glyphs, plain-language labels and the
wardrobe drawing (SVG). Pages import from here so every screen looks the same."""

from __future__ import annotations

import functools
import html as _html

import numpy as np
import streamlit as st

from fold_and_place import config
from fold_and_place.folding.templates import outline
from fold_and_place.garments.record import Garment, make_garment

STEPS = [
    ("wardrobe", "1", "Your wardrobe"),
    ("clothes", "2", "Your clothes"),
    ("plan", "3", "Your plan"),
    ("sharing", "4", "Sharing fairly"),
]
PALETTE = ["#3E7CB1", "#D9822B", "#4E9A6B", "#9B5DE5", "#D1495B", "#2A9D8F"]
WEAR_OPTIONS = {"Rarely": 0.2, "A few times a year": 0.5, "Monthly": 1.0, "Every 2 weeks": 2.0, "Weekly": 4.0,
                "Twice a week": 8.0, "Almost daily": 20.0}
VALUE_WORDS = {1: "not much", 2: "not much", 3: "a little", 4: "some", 5: "some", 6: "quite a bit", 7: "a lot",
               8: "a lot", 9: "treasured", 10: "treasured"}

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.fp-steps {display:flex; gap:.5rem; margin: 0 0 1.1rem 0; flex-wrap: wrap;}
.fp-step {display:flex; align-items:center; gap:.45rem; padding:.35rem .8rem .35rem .4rem; border-radius:999px;
  background:#F3ECE2; color:#7A6A58; font-size:.88rem; border:1px solid transparent;}
.fp-step b {display:inline-flex; width:1.45rem; height:1.45rem; border-radius:50%; align-items:center;
  justify-content:center; background:#E4D9CA; color:#5A4A3A; font-size:.8rem;}
.fp-step.on {background:#fff; color:#2B2420; border-color:#B4532A; font-weight:600;}
.fp-step.on b {background:#B4532A; color:#fff;}
.fp-step.done b {background:#4E9A6B; color:#fff;}
.fp-lead {font-size:1.12rem; color:#5A4A3A; margin:-.4rem 0 1.2rem 0; line-height:1.5;}
.fp-big {display:flex; gap:1rem; flex-wrap:wrap; margin:.2rem 0 1rem 0;}
.fp-num {flex:1 1 180px; background:#fff; border:1px solid #E4D9CA; border-radius:14px; padding:.9rem 1.1rem;}
.fp-num .v {font-size:1.9rem; font-weight:700; color:#2B2420; line-height:1.1;}
.fp-num .l {font-size:.9rem; color:#7A6A58; margin-top:.2rem;}
.fp-num.good .v {color:#2F7D52;} .fp-num.bad .v {color:#B0413E;}
.fp-card {background:#fff; border:1px solid #E4D9CA; border-radius:14px; padding:.8rem .9rem; height:100%;}
.fp-chip {display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.78rem; color:#fff; margin-right:.25rem;}
.fp-soft {color:#7A6A58; font-size:.88rem;}
.fp-verdict {border-radius:14px; padding:.9rem 1rem; border:1px solid #E4D9CA; background:#fff; height:100%;}
.fp-verdict h4 {margin:0 0 .25rem 0; font-size:1.02rem;}
.fp-verdict .how {color:#7A6A58; font-size:.86rem; min-height:2.6em;}
.fp-tag {display:inline-block; font-size:.78rem; padding:.12rem .5rem; border-radius:6px; margin:.15rem .25rem 0 0;}
.fp-tag.ok {background:#E3F1E8; color:#2F7D52;} .fp-tag.no {background:#F8E1DF; color:#B0413E;}
.fp-tag.mid {background:#FBF0DB; color:#946200;}
.fp-hero {font-size:2.6rem; font-weight:750; line-height:1.1; margin:.4rem 0 .6rem 0;}
.fp-hero-sub {font-size:1.2rem; color:#5A4A3A; max-width:46rem; line-height:1.55;}
div[data-testid="stExpander"] details summary p {font-weight:600;}
</style>
"""


def raw(html: str, where=None) -> None:
    """Render trusted HTML/SVG (st.html strips SVG, so this goes through markdown)."""
    (where or st).markdown(html.replace("\n", " "), unsafe_allow_html=True)


def setup(title: str, step: str | None = None, lead: str | None = None) -> None:
    from fold_and_place.app import state

    state.init()
    st.markdown(CSS, unsafe_allow_html=True)
    if step:
        order = [s[0] for s in STEPS]
        k = order.index(step)
        chips = []
        for i, (key, n, name) in enumerate(STEPS):
            cls = "on" if i == k else ("done" if state.step_done(key) else "")
            chips.append(f'<span class="fp-step {cls}"><b>{n if cls != "done" else "✓"}</b>{name}</span>')
        raw(f'<div class="fp-steps">{"".join(chips)}</div>')
    st.title(title)
    if lead:
        raw(f'<div class="fp-lead">{lead}</div>')


def big_numbers(items: list[tuple[str, str, str]]) -> None:
    """[(value, label, tone)] with tone in {'', 'good', 'bad'}."""
    cells = "".join(f'<div class="fp-num {t}"><div class="v">{_html.escape(v)}</div><div class="l">{_html.escape(lab)}</div></div>' for v, lab, t in items)
    raw(f'<div class="fp-big">{cells}</div>')


def next_button(label: str, page: str, key: str | None = None) -> None:
    if st.button(label, type="primary", key=key or f"next_{page}"):
        st.switch_page(page)


# ---- people -----------------------------------------------------------------------------
def person_label(key: str) -> str:
    people = st.session_state.get("people", {})
    if key in people:
        return people[key]["label"]
    p = config.personas().get(key)
    return p["label"] if p else (key or "Unassigned")


def person_colour(key: str) -> str:
    people = list(st.session_state.get("people", {}))
    if key in people:
        return PALETTE[people.index(key) % len(PALETTE)]
    return "#9A8C7C"


def chip(key: str) -> str:
    return f'<span class="fp-chip" style="background:{person_colour(key)}">{_html.escape(person_label(key))}</span>'


def wear_words(f: float) -> str:
    for name, v in sorted(WEAR_OPTIONS.items(), key=lambda t: -t[1]):
        if f >= v * 0.75:
            return name.lower()
    return "rarely"


def nearest_wear(f: float) -> str:
    return min(WEAR_OPTIONS, key=lambda k: abs(np.log((WEAR_OPTIONS[k] + 0.05) / (f + 0.05))))


def value_dots(v: float) -> str:
    n = int(round(v / 2))
    return "●" * n + "○" * (5 - n)


# ---- garment glyphs -----------------------------------------------------------------------
@functools.lru_cache(maxsize=512)
def _glyph_path(key: tuple) -> tuple[str, float]:
    pts = np.asarray(key, float)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    s = 1.0 / max(x1 - x0, y1 - y0, 1e-9)
    w = (x1 - x0) * s
    p = [((x - x0) * s + (1 - w) / 2, 1 - (y - y0) * s) for x, y in pts]
    h = (y1 - y0) * s
    p = [(x, y - (1 - h) / 2) for x, y in p]
    return "M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in p) + " Z", h


def glyph(g: Garment, size: int = 44, colour: str | None = None) -> str:
    """Little SVG of the garment's flat outline (its photographed outline when it has one)."""
    if g.outline:
        key = tuple(g.outline)
    else:
        key = _template_key(g.category)
    path, _ = _glyph_path(key)
    col = colour or person_colour(g.owner)
    return (f'<svg width="{size}" height="{size}" viewBox="-0.08 -0.08 1.16 1.16" xmlns="http://www.w3.org/2000/svg">'
            f'<path d="{path}" fill="{col}" fill-opacity="0.88" stroke="#2B2420" stroke-opacity="0.35" stroke-width="0.025"/></svg>')


@functools.lru_cache(maxsize=64)
def _template_key(category: str) -> tuple:
    g = make_garment(category, "glyph")
    return tuple(map(tuple, np.round(np.asarray(outline(g).exterior.coords)[:-1], 2)))


def thumb(g: Garment, size: int = 64) -> str:
    """Photo thumbnail when the garment came from a photo, else its glyph."""
    if g.image and g.image.startswith("data:image"):
        return f'<img src="{g.image}" style="width:{size}px;height:{size}px;object-fit:cover;border-radius:10px;border:1px solid #E4D9CA"/>'
    return glyph(g, size)


# ---- compartments in plain words --------------------------------------------------------------
def compartment_name(c, w) -> str:
    lo, hi = config.golden_zone()
    same = [x for x in w.compartments if x.kind == c.kind]
    same.sort(key=lambda x: (-x.z, x.x))
    n = same.index(c) + 1 if len(same) > 1 else None
    base = {"loft": "Loft", "rail_long": "Long hanging", "rail_short": "Short hanging", "shelf": "Shelf", "drawer": "Drawer"}[c.kind]
    if n:
        base += f" {n}"
    return base


def compartment_reach(c) -> str:
    z = c.zone_cost()
    if c.kind == "loft":
        return "needs a stool"
    return {1.0: "easy reach", 1.5: "a short bend or stretch"}.get(z, "bend down low")


# ---- the wardrobe drawing ------------------------------------------------------------------------
def wardrobe_svg(w, plan=None, *, selected: str | None = None, highlight: str | None = None, width: int = 460) -> str:
    """Front view: carcass, golden zone, compartments with plain names, and (with a plan) every
    garment drawn where it goes. Hover anything for details."""
    pad_l, pad_r, pad_t, pad_b = 8, 92, 10, 26
    s = (width - pad_l - pad_r) / w.width
    H = w.height * s + pad_t + pad_b
    X = lambda x: pad_l + x * s  # noqa: E731
    Y = lambda z: pad_t + (w.height - z) * s  # noqa: E731
    lo, hi = config.golden_zone()
    out = [f'<svg viewBox="0 0 {width} {H:.0f}" width="100%" xmlns="http://www.w3.org/2000/svg" style="max-width:{width}px;font-family:system-ui,sans-serif">']
    out.append('<defs><linearGradient id="wood" x1="0" x2="1"><stop offset="0" stop-color="#9C7650"/><stop offset="1" stop-color="#7E5C3C"/></linearGradient></defs>')
    out.append(f'<rect x="{X(0) - 6:.1f}" y="{Y(w.height) - 6:.1f}" width="{w.width * s + 12:.1f}" height="{w.height * s + 12:.1f}" rx="6" fill="url(#wood)"/>')
    out.append(f'<rect x="{X(0):.1f}" y="{Y(w.height):.1f}" width="{w.width * s:.1f}" height="{w.height * s:.1f}" fill="#E9DCC6"/>')
    # golden zone marker on the right
    gx = X(w.width) + 14
    hi = min(hi, w.height)
    out.append(f'<rect x="{gx:.1f}" y="{Y(hi):.1f}" width="8" height="{max(hi - lo, 0) * s:.1f}" rx="4" fill="#F2C14E"/>')
    out.append(f'<text x="{gx + 13:.1f}" y="{Y((lo + hi) / 2) - 4:.1f}" font-size="11" fill="#7A5A00">easy</text>'
               f'<text x="{gx + 13:.1f}" y="{Y((lo + hi) / 2) + 9:.1f}" font-size="11" fill="#7A5A00">reach</text>')
    out.append(f'<text x="{gx - 2:.1f}" y="{Y(w.height) - 1:.1f}" font-size="10" fill="#9A8C7C">{w.height:.0f} cm</text>')
    out.append(f'<text x="{gx + 13:.1f}" y="{Y(0):.1f}" font-size="10" fill="#9A8C7C">floor</text>')
    by_comp: dict[str, list] = {}
    if plan is not None:
        for lane in plan.lanes:
            by_comp.setdefault(lane.compartment, []).append(lane)
    for c in w.compartments:
        sel = c.id == selected
        x, y, cw, ch = X(c.x) + 1.5, Y(c.z + c.h) + 1.5, c.w * s - 3, c.h * s - 3
        fill = "#FBF7F0" if c.kind != "drawer" else "#F4ECDF"
        out.append(f'<g><title>{_html.escape(compartment_name(c, w))}: {compartment_reach(c)}</title>'
                   f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" rx="3" fill="{fill}" '
                   f'stroke="{"#B4532A" if sel else "#C9B79F"}" stroke-width="{3 if sel else 1}"/></g>')
        if c.kind.startswith("rail"):
            out.append(f'<line x1="{x + 3:.1f}" y1="{y + 7:.1f}" x2="{x + cw - 3:.1f}" y2="{y + 7:.1f}" stroke="#8A8F98" stroke-width="3" stroke-linecap="round"/>')
        if c.kind == "drawer":
            out.append(f'<rect x="{x + cw / 2 - 12:.1f}" y="{y + 5:.1f}" width="24" height="4" rx="2" fill="#9C7650"/>')
        lanes = by_comp.get(c.id, [])
        if lanes:
            out.extend(_items_svg(plan, c, lanes, x, y, cw, ch, s, highlight))
        lab = compartment_name(c, w)
        if cw > 44 and ch > 16:
            out.append(f'<text x="{x + 5:.1f}" y="{y + ch - 5:.1f}" font-size="10.5" fill="#7A6A58" opacity="0.9">{_html.escape(lab)}</text>')
    out.append("</svg>")
    return "".join(out)


def _items_svg(plan, c, lanes, x, y, cw, ch, s, highlight):
    from fold_and_place.storage.plan import IN, stack_width

    out = []

    def tip(gid):
        g, o = plan.garments[gid], plan.choice[gid]
        return f"<title>{_html.escape(person_label(g.owner))}: {_html.escape(g.label)} ({gid}) - {_html.escape(o.method)}</title>"

    if lanes[0].layout == "rail":
        px = x + 4
        for gid in lanes[0].items:
            g, o = plan.garments[gid], plan.choice[gid]
            wd = max(2.0, o.rod_in * IN * s)
            drop = min(o.drop * s, ch - 12)
            col = person_colour(g.owner)
            stroke = ' stroke="#B4532A" stroke-width="2.5"' if gid == highlight else ' stroke="#fff" stroke-width="0.6"'
            out.append(f'<g>{tip(gid)}<path d="M{px + wd / 2:.1f},{y + 7:.1f} l0,4" stroke="#6B6F76" stroke-width="1"/>'
                       f'<rect x="{px:.1f}" y="{y + 11:.1f}" width="{wd * 0.92:.1f}" height="{drop:.1f}" rx="2" fill="{col}" fill-opacity="0.9"{stroke}/></g>')
            px += wd
    elif lanes[0].layout == "stack":
        px = x + 2
        for lane in lanes:
            lw = max(stack_width(plan.choice[g], c) or 0 for g in lane.items) * s
            pz = y + ch - 2
            total = sum(plan.choice[g].h for g in lane.items) * s
            f = min(1.0, (ch - 6) / max(total, 1e-9))  # thin packets: never draw past the shelf
            for gid in lane.items:
                g, o = plan.garments[gid], plan.choice[gid]
                hgt = o.h * s * f
                stroke = ' stroke="#B4532A" stroke-width="2.5"' if gid == highlight else (' stroke="#fff" stroke-width="0.5"' if hgt > 3 else "")
                out.append(f'<g>{tip(gid)}<rect x="{px:.1f}" y="{pz - hgt:.1f}" width="{max(lw - 1.5, 2):.1f}" height="{hgt:.1f}" rx="1.5" fill="{person_colour(g.owner)}" fill-opacity="0.9"{stroke}/></g>')
                pz -= hgt
            px += lw
    else:  # drawer rows seen from the front: packets standing on edge, back rows paler
        n = len(lanes)
        for k, lane in enumerate(reversed(lanes)):
            px = x + 3
            for gid in lane.items:
                g, o = plan.garments[gid], plan.choice[gid]
                wd = max(1.6, o.h * s)
                hgt = min(o.b * s, ch - 14) * (0.75 + 0.25 * (k + 1) / n)
                stroke = ' stroke="#B4532A" stroke-width="2.5"' if gid == highlight else ' stroke="#fff" stroke-width="0.4"'
                op = 0.45 + 0.5 * (k + 1) / n
                out.append(f'<g>{tip(gid)}<rect x="{px:.1f}" y="{y + ch - 2 - hgt:.1f}" width="{wd:.1f}" height="{hgt:.1f}" fill="{person_colour(g.owner)}" fill-opacity="{op:.2f}"{stroke}/></g>')
                px += wd
    return out


def legend(people: list[str]) -> str:
    return " ".join(chip(p) for p in people)
