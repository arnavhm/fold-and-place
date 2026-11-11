"""Geometric fold engine ("g-folds").

A garment lying flat is a set of polygon layers. A fold along an axis-aligned line reflects every
layer's part on the moving side across the line and stacks it on top (in reverse order, because the
flap turns over). Folded size, layer count, thickness, the text steps and the animation all come
from the same step list, so they cannot disagree.

Model after van den Berg, Miller, Goldberg & Abbeel, "Gravity-based robotic cloth folding" (2010):
cloth lying flat is a polygon, folds are reflections across a line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import shapely
from shapely.affinity import scale
from shapely.geometry.base import BaseGeometry

from fold_and_place.folding.templates import outline
from fold_and_place.garments.record import Garment

BIG = 1e5
EPS = 1e-6
Axis = Literal["x", "y"]
Move = Literal["lt", "gt"]


@dataclass(frozen=True)
class FoldStep:
    """Fold along the line ``axis == coord``; the part on side ``move`` turns over onto the rest.

    axis "x": vertical line x = coord.   move "lt": the part with x < coord moves.
    axis "y": horizontal line y = coord. move "lt": the part with y < coord moves (hem goes up).
    """

    axis: Axis
    coord: float
    move: Move
    text: str = ""

    def to_dict(self) -> dict:
        return {"axis": self.axis, "coord": round(self.coord, 3), "move": self.move, "text": self.text}


@dataclass
class Layer:
    geom: BaseGeometry
    weight: int  # how many fabric plies this polygon stands for (2 = front + back)


@dataclass
class FoldState:
    layers: list[Layer]
    steps: list[FoldStep] = field(default_factory=list)

    @classmethod
    def from_garment(cls, g: Garment) -> FoldState:
        return cls([Layer(outline(g), g.base_layers)])

    # ----- geometry summaries ----------------------------------------------------------
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        b = np.array([lay.geom.bounds for lay in self.layers])
        return float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())

    @property
    def width(self) -> float:
        x0, _, x1, _ = self.bounds
        return x1 - x0

    @property
    def length(self) -> float:
        _, y0, _, y1 = self.bounds
        return y1 - y0

    def ply_area(self) -> float:
        """Total fabric area x plies. Folding never changes it (area conservation)."""
        return float(sum(lay.geom.area * lay.weight for lay in self.layers))

    def ply_grid(self, n: int = 90) -> np.ndarray:
        """Number of fabric plies at each point of an n x n grid over the footprint."""
        x0, y0, x1, y1 = self.bounds
        xs = np.linspace(x0, x1, n + 1)[:-1] + (x1 - x0) / (2 * n)
        ys = np.linspace(y0, y1, n + 1)[:-1] + (y1 - y0) / (2 * n)
        X, Y = np.meshgrid(xs, ys)
        count = np.zeros_like(X)
        for lay in self.layers:
            count += lay.weight * shapely.contains_xy(lay.geom, X, Y)
        return count

    def max_plies(self) -> int:
        return int(self.ply_grid().max())


def _clip(geom: BaseGeometry, axis: Axis, coord: float, side: Move) -> BaseGeometry:
    if axis == "x":
        box = (-BIG, -BIG, coord, BIG) if side == "lt" else (coord, -BIG, BIG, BIG)
    else:
        box = (-BIG, -BIG, BIG, coord) if side == "lt" else (-BIG, coord, BIG, BIG)
    part = shapely.clip_by_rect(geom, *box)
    # keep only the polygonal pieces (clipping can leave stray lines on the fold line)
    polys = [p for p in shapely.get_parts(part) if p.geom_type == "Polygon" and p.area > EPS]
    if not polys:
        return shapely.Polygon()
    return polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys)


def _reflect(geom: BaseGeometry, axis: Axis, coord: float) -> BaseGeometry:
    if axis == "x":
        return scale(geom, xfact=-1.0, yfact=1.0, origin=(coord, 0.0))
    return scale(geom, xfact=1.0, yfact=-1.0, origin=(0.0, coord))


def apply_fold(state: FoldState, step: FoldStep) -> FoldState:
    """Return a new state with ``step`` applied. Moving layers flip and land on top, reversed."""
    stay_side: Move = "gt" if step.move == "lt" else "lt"
    stays: list[Layer] = []
    moving: list[Layer] = []
    for lay in state.layers:
        s = _clip(lay.geom, step.axis, step.coord, stay_side)
        m = _clip(lay.geom, step.axis, step.coord, step.move)
        if not s.is_empty:
            stays.append(Layer(s, lay.weight))
        if not m.is_empty:
            moving.append(Layer(_reflect(m, step.axis, step.coord), lay.weight))
    if not moving:  # the line misses the cloth: not a real fold
        return state
    return FoldState(stays + moving[::-1], state.steps + [step])


def run(g: Garment, steps: list[FoldStep]) -> FoldState:
    st = FoldState.from_garment(g)
    for s in steps:
        st = apply_fold(st, s)
    return st


# ----- fold primitives used to build named methods ----------------------------------------


def sides_in(state: FoldState, keep_w: float, max_iter: int = 8) -> FoldState:
    """Fold both sides in so a centred panel ``keep_w`` wide remains; long sleeves are folded back
    along the edge until nothing sticks out (the department-store / board fold)."""
    x0, _, x1, _ = state.bounds
    cx = (x0 + x1) / 2
    left, right = cx - keep_w / 2, cx + keep_w / 2
    st = state
    if st.bounds[0] < left - EPS:
        st = apply_fold(st, FoldStep("x", left, "lt", f"Fold the left side (and sleeve) in along a line {keep_w / 2:.1f} cm left of centre."))
    if st.bounds[2] > right + EPS:
        st = apply_fold(st, FoldStep("x", right, "gt", f"Fold the right side (and sleeve) in along a line {keep_w / 2:.1f} cm right of centre."))
    for _ in range(max_iter):
        bx0, _, bx1, _ = st.bounds
        if bx1 > right + 1e-3:
            st = apply_fold(st, FoldStep("x", right, "gt", "Fold the sleeve that sticks out on the right back along the edge."))
        elif bx0 < left - 1e-3:
            st = apply_fold(st, FoldStep("x", left, "lt", "Fold the sleeve that sticks out on the left back along the edge."))
        else:
            break
    return st


def panels(state: FoldState, axis: Axis, n: int, what: str = "") -> FoldState:
    """Wrap-fold into ``n`` equal panels along ``axis`` (n = 2 halves, 3 thirds, ...).

    Each fold turns the stack that has built up at the low end over onto the next panel.
    """
    if n <= 1:
        return state
    x0, y0, x1, y1 = state.bounds
    lo, hi = (x0, x1) if axis == "x" else (y0, y1)
    p = (hi - lo) / n
    st = state
    word = {2: "half", 3: "thirds", 4: "quarters"}.get(n, f"{n} panels")
    for k in range(1, n):
        c = lo + k * p
        if axis == "y":
            txt = f"Fold the bottom up onto the next panel ({word} of the length; panel {p:.1f} cm)."
        else:
            txt = f"Fold the left edge over onto the next panel ({word} of the width; panel {p:.1f} cm)."
        if what:
            txt = f"{what}: {txt}"
        st = apply_fold(st, FoldStep(axis, c, "lt", txt))
    return st


def fit_length(state: FoldState, target: float) -> FoldState:
    """Board-style length fold: turn the hem up by just enough, then fold into panels of exactly
    ``target`` length. This is how a folding board works: the packet ends up the board's length."""
    _, y0, _, y1 = state.bounds
    length = y1 - y0
    if length <= target + EPS:
        return state
    n = math.ceil(length / target - 1e-9)
    e = length - (n - 1) * target  # hem turn-up, 0 < e <= target
    st = state
    if e < target - EPS:
        st = apply_fold(st, FoldStep("y", y0 + e, "lt", f"Turn the hem up by {e:.1f} cm."))
    for _ in range(1, n - 1 if e < target - EPS else n):
        c = st.bounds[1] + target
        st = apply_fold(st, FoldStep("y", c, "lt", f"Fold the bottom up to make a {target:.1f} cm panel."))
    return st


def leg_over_leg(state: FoldState) -> FoldState:
    x0, _, x1, _ = state.bounds
    cx = (x0 + x1) / 2
    return apply_fold(state, FoldStep("x", cx, "lt", "Fold one leg over the other along the centre seam."))
