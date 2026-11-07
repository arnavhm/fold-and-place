"""Matplotlib drawings for the dashboard: wardrobe front view (with golden zone and placed items)
and step-by-step fold diagrams (the 2D fallback for the 3D animation)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from fold_and_place import config  # noqa: E402
from fold_and_place.folding.engine import FoldState, apply_fold  # noqa: E402
from fold_and_place.storage.plan import IN, Plan, stack_width  # noqa: E402
from fold_and_place.storage.wardrobe import Wardrobe  # noqa: E402

KIND_FILL = {"shelf": "#f4efe6", "loft": "#ece6da", "drawer": "#efe9df", "rail_long": "#f7f4ee", "rail_short": "#f7f4ee"}
OWNER = {"student": "#8fa9c9", "mother": "#d9a75f", "father": "#9cc39c"}


def wardrobe_front(w: Wardrobe, plan: Plan | None = None, highlight: str | None = None):
    fig, ax = plt.subplots(figsize=(max(4.0, (w.width + 30) / 32), max(5.0, w.height / 32)))
    lo, hi = config.golden_zone()
    ax.add_patch(Rectangle((-6, lo), w.width + 12, hi - lo, color="#fff4c2", alpha=0.6, zorder=0))
    ax.text(w.width + 7, (lo + hi) / 2, "golden\nzone", fontsize=7, va="center", color="#8a6d00")
    ax.add_patch(Rectangle((0, 0), w.width, w.height, fill=False, lw=2, ec="#5a4630"))
    for c in w.compartments:
        ax.add_patch(Rectangle((c.x, c.z), c.w, c.h, fc=KIND_FILL[c.kind], ec="#8a7660", lw=1, alpha=0.9, zorder=1))
        if c.kind.startswith("rail"):
            ax.plot([c.x + 1, c.x + c.w - 1], [c.z + c.h - 2] * 2, color="#6b6b6b", lw=2.5, zorder=3)
        ax.text(c.x + 1.5, c.z + c.h - 4, c.id, fontsize=6.5, color="#5a4630", zorder=5)
    if plan is not None:
        _draw_items(ax, plan, highlight)
    ax.set_xlim(-8, w.width + 22)
    ax.set_ylim(-4, w.height + 6)
    ax.set_aspect("equal")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_ylabel("height above floor (cm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(w.label, fontsize=9)
    fig.tight_layout()
    return fig


def _draw_items(ax, plan: Plan, highlight: str | None):
    by_comp: dict[str, list] = {}
    for lane in plan.lanes:
        by_comp.setdefault(lane.compartment, []).append(lane)
    for cid, lanes in by_comp.items():
        c = plan.wardrobe.get(cid)
        if lanes[0].layout == "rail":
            x = c.x + 1
            for gid in lanes[0].items:
                o, g = plan.choice[gid], plan.garments[gid]
                wdt = o.rod_in * IN
                drop = min(o.drop, c.h - 3)
                ax.add_patch(Rectangle((x, c.z + c.h - 3 - drop), wdt * 0.9, drop, fc=OWNER.get(g.owner, "#bbb"), ec="white", lw=0.4, zorder=4, alpha=0.95 if gid != highlight else 1))
                if gid == highlight:
                    ax.add_patch(Rectangle((x, c.z + c.h - 3 - drop), wdt * 0.9, drop, fill=False, ec="red", lw=2, zorder=6))
                x += wdt
        elif lanes[0].layout == "stack":
            x = c.x + 0.5
            for lane in lanes:
                lw = max(stack_width(plan.choice[g], c) or 0 for g in lane.items)
                z = c.z
                for gid in lane.items:
                    o, g = plan.choice[gid], plan.garments[gid]
                    ax.add_patch(Rectangle((x, z), lw, o.h, fc=OWNER.get(g.owner, "#bbb"), ec="white", lw=0.3, zorder=4))
                    if gid == highlight:
                        ax.add_patch(Rectangle((x, z), lw, o.h, fill=False, ec="red", lw=2, zorder=6))
                    z += o.h
                x += lw
        else:  # drawer rows seen from the front: each row's packets side by side, standing
            x0 = c.x + 0.5
            n = len(lanes)
            for k, lane in enumerate(lanes):
                x = x0
                for gid in lane.items:
                    o, g = plan.choice[gid], plan.garments[gid]
                    hgt = min(o.b, c.h) * (1 - 0.15 * k)
                    ax.add_patch(Rectangle((x, c.z), o.h, hgt, fc=OWNER.get(g.owner, "#bbb"), ec="white", lw=0.3, zorder=4 + n - k, alpha=1 - 0.25 * k))
                    if gid == highlight:
                        ax.add_patch(Rectangle((x, c.z), o.h, hgt, fill=False, ec="red", lw=2, zorder=9))
                    x += o.h


def fold_steps_figure(g, steps, cols: int = 4):
    """One small panel per state: flat, then after each fold (darker = higher layers)."""
    st = FoldState.from_garment(g)
    states = [st]
    for s in steps:
        st = apply_fold(st, s)
        states.append(st)
    n = len(states)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        s = states[i]
        for k, lay in enumerate(s.layers):
            for part in shapely.get_parts(lay.geom):
                ax.add_patch(MplPolygon(np.asarray(part.exterior.coords), closed=True, fc=cmap(k / max(1, len(s.layers) - 1)), ec="black", lw=0.5, alpha=0.4))
        if i > 0:
            step = steps[i - 1]
            x0, y0, x1, y1 = states[i - 1].bounds
            if step.axis == "x":
                ax.plot([step.coord] * 2, [y0, y1], "r--", lw=1)
            else:
                ax.plot([x0, x1], [step.coord] * 2, "r--", lw=1)
        x0, y0, x1, y1 = s.bounds
        ax.set_xlim(x0 - 4, x1 + 4)
        ax.set_ylim(y0 - 4, y1 + 4)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.set_title("flat" if i == 0 else f"after fold {i}: {s.width:.0f} x {s.length:.0f} cm", fontsize=8)
    fig.tight_layout()
    return fig
