"""E1 - Are fold sizes believable?

1. Calibrate the loft factor against a published retail tee fold (11 x 9 x 0.5 in).
2. Fold tops with a 23 cm board width and compare to the 23 x 30.5 cm department-store board,
   for the board fold (hem turned up) and a generic equal-panel fold.
3. Check area conservation over random fold sequences for every category.
4. Summarise how many fold options each category gets, and the most compact one.
5. Draw flat vs folded diagrams for a shirt, jeans and a saree.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.patches import Polygon as MplPolygon

from fold_and_place import config
from fold_and_place.experiments.common import out_dir, save_csv, save_fig, write_notes
from fold_and_place.folding import FoldState, FoldStep, apply_fold, generate_options
from fold_and_place.folding.calibration import board_check, board_fold, calibrate_k_loft
from fold_and_place.folding.engine import panels
from fold_and_place.garments import make_garment


def _area_check(seed: int = 0, trials: int = 20, folds: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cat in config.categories():
        g = make_garment(cat, cat)
        worst = 0.0
        for _ in range(trials):
            st0 = FoldState.from_garment(g)
            st = st0
            for _ in range(folds):
                axis = "x" if rng.random() < 0.5 else "y"
                x0, y0, x1, y1 = st.bounds
                lo, hi = (x0, x1) if axis == "x" else (y0, y1)
                c = lo + (hi - lo) * rng.uniform(0.15, 0.85)
                st = apply_fold(st, FoldStep(axis, c, "lt" if rng.random() < 0.5 else "gt"))
            worst = max(worst, abs(st.ply_area() - st0.ply_area()) / st0.ply_area())
        rows.append({"category": cat, "trials": trials, "folds_per_trial": folds, "max_rel_area_error": worst})
    return pd.DataFrame(rows)


def _option_summary() -> pd.DataFrame:
    rows = []
    for cat, c in config.categories().items():
        g = make_garment(cat, cat, "M")
        opts = generate_options(g)
        packets = [o for o in opts if o.mode != "hang"]
        best = min(packets, key=lambda o: o.volume) if packets else None
        rows.append(
            {
                "category": cat,
                "group": c["group"],
                "flat_w_cm": g.flat_w,
                "flat_l_cm": g.flat_l,
                "options": len(opts),
                "modes": ",".join(sorted({o.mode for o in opts})),
                "most_compact": best.method if best else "hang only",
                "compact_a_cm": round(best.a, 1) if best else None,
                "compact_b_cm": round(best.b, 1) if best else None,
                "compact_h_cm": round(best.h, 2) if best else None,
            }
        )
    return pd.DataFrame(rows)


def _draw_state(ax, st: FoldState, title: str) -> None:
    cmap = plt.get_cmap("viridis")
    n = len(st.layers)
    for i, lay in enumerate(st.layers):
        for part in shapely.get_parts(lay.geom):
            xy = np.asarray(part.exterior.coords)
            ax.add_patch(MplPolygon(xy, closed=True, fc=cmap(i / max(1, n - 1)), ec="black", lw=0.6, alpha=0.35))
    x0, y0, x1, y1 = st.bounds
    ax.set_xlim(x0 - 5, x1 + 5)
    ax.set_ylim(y0 - 5, y1 + 5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def _diagrams(d):
    shirt = make_garment("shirt", "shirt", "L")
    jeans = make_garment("jeans", "jeans", "M")
    saree = make_garment("saree", "saree")
    from fold_and_place.folding.engine import leg_over_leg

    cases = [
        (shirt, board_fold(shirt, 23.0, 30.5), "Shirt L: board fold"),
        (jeans, panels(leg_over_leg(FoldState.from_garment(jeans)), "y", 3), "Jeans: leg over leg, thirds"),
        (saree, panels(panels(FoldState.from_garment(saree), "x", 4), "y", 16), "Saree: lengthwise x4, 16 panels"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), constrained_layout=True)
    for col, (g, st, title) in enumerate(cases):
        _draw_state(axes[0, col], FoldState.from_garment(g), f"{g.label} flat ({g.flat_w:.0f} x {g.flat_l:.0f} cm)")
        _draw_state(
            axes[1, col], st, f"{title}\n{st.width:.1f} x {st.length:.1f} cm, {st.max_plies()} plies, {len(st.steps)} folds"
        )
    fig.suptitle("E1: flat outline (top) vs folded packet (bottom); darker = later layers", fontsize=10)
    return save_fig(fig, d, "E1_fold_diagrams.png")


def run(path=None) -> dict:
    d = out_dir(path)
    cal = calibrate_k_loft()
    cal_df = pd.DataFrame([cal])
    save_csv(cal_df, d, "E1_calibration.csv")

    board_rows = []
    for cat, size in (("shirt", "L"), ("tshirt", "L"), ("tshirt", "M"), ("kurta_men", "L"), ("polo", "L")):
        board_rows += board_check(cat, size)
    board_df = pd.DataFrame(board_rows)
    save_csv(board_df, d, "E1_board.csv")

    area_df = _area_check()
    save_csv(area_df, d, "E1_area.csv")

    opt_df = _option_summary()
    save_csv(opt_df, d, "E1_options.csv")
    _diagrams(d)

    eq = board_df[board_df["method"] == "equal panels"]
    bd = board_df[board_df["method"].str.startswith("board")]
    summary = {
        "k_loft": round(cal["k_loft"], 3),
        "k_loft_in_config": config.k_loft(),
        "board_fold_max_abs_err_pct": float(bd[["err_w_pct", "err_l_pct"]].abs().max().max()),
        "equal_panel_len_err_pct_range": (float(eq["err_l_pct"].min()), float(eq["err_l_pct"].max())),
        "max_area_error": float(area_df["max_rel_area_error"].max()),
        "categories": len(opt_df),
        "mean_options_per_category": round(float(opt_df["options"].mean()), 1),
    }
    write_notes(
        d,
        "E1_notes.md",
        f"""
# E1 - Fold sizes

- **Loft factor** calibrated on {cal['garment']}: {cal['plies']} plies x {cal['layer_thickness_mm']} mm
  -> published 0.5 in (1.27 cm) stack gives **k_loft = {cal['k_loft']:.3f}** (config holds {config.k_loft()}).
- **Board fold** (sides in to 23 cm, hem turned up, folded to 30.5 cm) lands within
  {summary['board_fold_max_abs_err_pct']:.1f}% of the 23 x 30.5 cm department-store board for every top tested.
  This checks the geometry (sleeve fold-backs, panel arithmetic); the board width is an input.
- **Equal-panel fold** with the same width misses the board length by
  {summary['equal_panel_len_err_pct_range'][0]:.1f}% to {summary['equal_panel_len_err_pct_range'][1]:.1f}%:
  equal thirds come out shorter than a board fold, which tucks the hem first.
- **Area conservation**: worst relative error over {len(area_df)} categories x 20 random 6-fold
  sequences = {summary['max_area_error']:.2e} (numerical noise only).
- **Options**: {summary['mean_options_per_category']} fold options per category on average
  (after dropping options that fit nowhere or are beaten on every dimension).

Limits: geometry only (no cloth physics); single-layer thicknesses are partly estimates
(see configs/fabrics.yaml `basis`); k_loft is calibrated on one published example.
""",
    )
    return summary


if __name__ == "__main__":
    print(run())
