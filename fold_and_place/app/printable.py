"""A printable, shelf-by-shelf wardrobe plan (one HTML file; print it or save as PDF)."""

from __future__ import annotations

import datetime as _dt
import html

from fold_and_place.app import ui

MODE = {"flat": "Fold, then stack", "file": "Fold, then stand on its edge", "roll": "Roll", "hang": "Hang"}


def plan_html(plan) -> str:
    w = plan.wardrobe
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Wardrobe plan</title><style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#2B2420;max-width:820px;margin:24px auto;padding:0 16px}"
        "h1{margin-bottom:0}.sub{color:#7A6A58;margin-top:4px}"
        "section{border:1px solid #E4D9CA;border-radius:12px;padding:12px 16px;margin:14px 0;page-break-inside:avoid}"
        "h2{font-size:1.1rem;margin:0 0 6px 0}.reach{color:#7A6A58;font-weight:400;font-size:.9rem}"
        "ol{margin:6px 0 0 0;padding-left:22px}li{margin:6px 0}.how{color:#5A4A3A;font-size:.9rem}"
        ".chip{display:inline-block;padding:0 8px;border-radius:999px;color:#fff;font-size:.8rem;margin-left:6px}"
        ".steps{font-size:.85rem;color:#5A4A3A;margin:2px 0 0 0}"
        "@media print{section{border-color:#bbb}}</style></head><body>",
        "<h1>Wardrobe plan</h1>",
        f"<div class='sub'>{html.escape(w.label)} · {len(plan.stored)} of {len(plan.garments)} clothes · "
        f"{_dt.date.today():%d %b %Y}</div>",
        f"<div style='margin:14px 0'>{ui.wardrobe_svg(w, plan, width=420)}</div>",
    ]
    for c in sorted(w.compartments, key=lambda c: (-c.z, c.x)):
        lanes = plan.lanes_in(c.id)
        items = [g for lane in lanes for g in lane.items]
        if not items:
            continue
        order = "front to back, left to right" if c.kind == "drawer" else ("left to right" if c.kind.startswith("rail") else "bottom to top, left to right")
        parts.append(f"<section><h2>{html.escape(ui.compartment_name(c, w))} <span class='reach'>· {ui.compartment_reach(c)} · {len(items)} items, {order}</span></h2><ol>")
        for gid in items:
            g, o = plan.garments[gid], plan.choice[gid]
            steps = "".join(f"<li>{html.escape(s.text)}</li>" for s in o.steps)
            parts.append(
                f"<li><b>{html.escape(g.label)}</b><span class='chip' style='background:{ui.person_colour(g.owner)}'>{html.escape(ui.person_label(g.owner))}</span>"
                f"<div class='how'>{MODE[o.mode]}: {html.escape(o.method)}</div>"
                + (f"<ol class='steps' type='a'>{steps}</ol>" if steps else "")
                + "</li>"
            )
        parts.append("</ol></section>")
    if plan.overflow:
        parts.append("<section><h2>Doesn't fit</h2><ol>" + "".join(
            f"<li>{html.escape(plan.garments[g].label)} ({html.escape(ui.person_label(plan.garments[g].owner))})</li>" for g in plan.overflow) + "</ol></section>")
    parts.append("<p class='sub'>Made with Fold &amp; Place.</p></body></html>")
    return "".join(parts)
