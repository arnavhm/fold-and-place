"""Wardrobe presets, compartments and how hard each spot is to reach."""

from __future__ import annotations

from dataclasses import dataclass, field

from fold_and_place import config

IN = 2.54

# Layout of a compartment: folded packets in side-by-side STACKS (shelves, loft), packets filed on
# edge in front-to-back ROWS (drawers), or hangers on a RAIL.
LAYOUT = {"shelf": "stack", "loft": "stack", "drawer": "row", "rail_long": "rail", "rail_short": "rail"}

# How awkward it is to take one item out, by storage mode. A flat stack hides everything under
# the top item; filed, rolled and hung items can be seen and pulled out directly.
MODE_FACTOR = {"flat": 1.3, "file": 1.0, "roll": 1.0, "hang": 1.0}


@dataclass(frozen=True)
class Compartment:
    id: str
    kind: str
    x: float
    z: float
    w: float
    h: float
    d: float

    @property
    def layout(self) -> str:
        return LAYOUT[self.kind]

    @property
    def centre_z(self) -> float:
        if self.layout == "rail":
            return self.z + self.h  # you reach for the hanger at the rod
        return self.z + self.h / 2

    @property
    def volume(self) -> float:
        return self.w * self.h * self.d

    def zone_cost(self, golden: tuple[float, float] | None = None) -> float:
        """1 = golden zone (waist-ish to shoulder); higher = bend, stretch or climb.

        Golden zone ~18 in below waist up to shoulder height (Racklify); default 55-135 cm.
        """
        lo, hi = golden or config.golden_zone()
        if self.kind == "loft":
            return 3.0
        c = self.centre_z
        if lo <= c <= hi:
            return 1.0
        if c > hi:
            return 1.5 if c <= hi + 40 else 3.0
        return 1.5 if c >= 30 else 3.0


@dataclass
class Wardrobe:
    name: str
    label: str
    width: float
    height: float
    depth: float
    compartments: list[Compartment] = field(default_factory=list)

    def get(self, cid: str) -> Compartment:
        for c in self.compartments:
            if c.id == cid:
                return c
        raise KeyError(cid)

    @property
    def packet_volume(self) -> float:
        return sum(c.volume for c in self.compartments if c.layout != "rail")

    @property
    def rod_cm(self) -> float:
        return sum(c.w for c in self.compartments if c.layout == "rail")


def load_preset(name: str) -> Wardrobe:
    p = config.presets()[name]
    comps = [Compartment(**c) for c in p["compartments"]]
    return Wardrobe(name, p["label"], p["width"], p["height"], p["depth"], comps)


# Shoulder heights set at run time (the dashboard's "who uses this wardrobe"); these win over
# configs/personas.yaml.
SHOULDER_CM: dict[str, float] = {}


def golden_for(owner: str) -> tuple[float, float]:
    """Golden zone for a household member: default bottom, their shoulder height on top."""
    lo, hi = config.golden_zone()
    if owner in SHOULDER_CM:
        return lo, float(SHOULDER_CM[owner])
    persona = config.personas().get(owner)
    if persona and "shoulder_cm" in persona:
        hi = float(persona["shoulder_cm"])
    return lo, hi
