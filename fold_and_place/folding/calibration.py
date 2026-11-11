"""Calibrate the loft factor and check fold sizes against published retail folds (experiment E1).

References
- Retail fold spec example: a T-shirt folded to 11 x 9 x 0.5 in (Racklify, "Retail fold").
- Department-store folding board for large shirts: 9 x 12 in / 23 x 30.5 cm (Make: magazine).
"""

from __future__ import annotations

import math

from fold_and_place.folding.engine import FoldState, fit_length, panels, sides_in
from fold_and_place.garments.record import Garment, make_garment

IN = 2.54
RETAIL_TEE_IN = (11.0, 9.0, 0.5)  # length, width, thickness
BOARD_SHIRT_CM = (30.5, 23.0)  # length, width


def board_fold(g: Garment, board_w_cm: float, board_l_cm: float) -> FoldState:
    """Department-store board fold: sides (and sleeves) in to the board width, hem turned up,
    then folded to exactly the board length."""
    return fit_length(sides_in(FoldState.from_garment(g), board_w_cm), board_l_cm)


def equal_panel_fold(g: Garment, board_w_cm: float, board_l_cm: float) -> FoldState:
    """Generic fold: same side folds, then the fewest EQUAL length panels that fit the board."""
    st = sides_in(FoldState.from_garment(g), board_w_cm)
    n = max(1, math.ceil(st.length / board_l_cm - 1e-9))
    return panels(st, "y", n)


def calibrate_k_loft(size: str = "M") -> dict:
    """k_loft = published stack height / (plies x single-layer thickness), retail tee fold."""
    g = make_garment("tshirt", "calib-tee", size)
    L, W, T = (v * IN for v in RETAIL_TEE_IN)
    st = board_fold(g, W, L)
    plies = st.max_plies()
    k = T / (plies * g.thickness_cm)
    return {
        "garment": f"T-shirt size {size} ({g.flat_w:.0f} x {g.flat_l:.0f} cm)",
        "plies": plies,
        "layer_thickness_mm": round(g.thickness_cm * 10, 3),
        "target_thickness_cm": round(T, 3),
        "k_loft": k,
        "folded_w_cm": st.width,
        "folded_l_cm": st.length,
        "target_w_cm": W,
        "target_l_cm": L,
        "steps": len(st.steps),
    }


def board_check(category: str = "shirt", size: str = "L") -> list[dict]:
    """Fold with the 23 cm board width and compare both length methods to the 23 x 30.5 cm board."""
    g = make_garment(category, f"board-{category}", size)
    bl, bw = BOARD_SHIRT_CM
    rows = []
    for name, fn in (("board fold (hem turned up)", board_fold), ("equal panels", equal_panel_fold)):
        st = fn(g, bw, bl)
        rows.append(
            {
                "garment": f"{g.label} size {size} ({g.flat_w:.0f} x {g.flat_l:.0f} cm)",
                "method": name,
                "folded_w_cm": round(st.width, 2),
                "folded_l_cm": round(st.length, 2),
                "target_w_cm": bw,
                "target_l_cm": bl,
                "err_w_pct": round(100 * (st.width - bw) / bw, 1),
                "err_l_pct": round(100 * (st.length - bl) / bl, 1),
                "plies": st.max_plies(),
                "steps": len(st.steps),
            }
        )
    return rows
