"""Everything the pages share: who uses the wardrobe, which wardrobe, the clothes, and the latest
plans. Kept in st.session_state; can be saved to / loaded from a JSON file."""

from __future__ import annotations

import dataclasses
import json
import os

import streamlit as st

from fold_and_place import config
from fold_and_place.garments import household_garments
from fold_and_place.garments.record import Garment
from fold_and_place.storage import load_preset
from fold_and_place.storage import wardrobe as wardrobe_mod
from fold_and_place.storage.wardrobe import Compartment, Wardrobe

SHOULDER_RATIO = 0.82  # shoulder height / standing height (adult anthropometry, rounded)


def demo_people() -> dict[str, dict]:
    out = {}
    for k, p in config.personas().items():
        out[k] = {"label": p["label"], "height_cm": round(p.get("shoulder_cm", 140) / SHOULDER_RATIO)}
    return out


def init() -> None:
    ss = st.session_state
    if "people" not in ss:
        load_demo()
    apply_people()


def load_demo() -> None:
    ss = st.session_state
    ss.people = demo_people()
    ss.preset = "two_door"
    ss.wardrobe = load_preset("two_door")
    ss.garments = household_garments()
    ss.mode = "demo"
    reset_plans()


def start_fresh(name: str = "Me", height: float = 170) -> None:
    ss = st.session_state
    ss.people = {"me": {"label": name, "height_cm": height}}
    ss.preset = "two_door"
    ss.wardrobe = load_preset("two_door")
    ss.garments = []
    ss.mode = "own"
    reset_plans()


def reset_plans() -> None:
    st.session_state.plans = {}
    st.session_state.plan = None
    st.session_state.pop("plan_key", None)


def apply_people() -> None:
    wardrobe_mod.SHOULDER_CM.clear()
    for k, p in st.session_state.people.items():
        wardrobe_mod.SHOULDER_CM[k] = float(p["height_cm"]) * SHOULDER_RATIO


def wardrobe() -> Wardrobe:
    init()
    return st.session_state.wardrobe


def garments() -> list[Garment]:
    init()
    return st.session_state.garments


def set_garments(gs: list[Garment]) -> None:
    st.session_state.garments = gs
    reset_plans()


def set_preset(name: str) -> None:
    st.session_state.preset = name
    st.session_state.wardrobe = load_preset(name)
    reset_plans()


def set_compartments(rows: list[dict]) -> None:
    w = st.session_state.wardrobe
    comps = [Compartment(**{k: (float(v) if k in "xzwhd" else str(v)) for k, v in r.items()}) for r in rows]
    st.session_state.wardrobe = dataclasses.replace(w, compartments=comps, name=w.name + "*")
    reset_plans()


def presets() -> dict[str, str]:
    return {k: v["label"] for k, v in config.presets().items()}


def step_done(step: str) -> bool:
    ss = st.session_state
    return {
        "wardrobe": ss.get("wardrobe_seen", False),
        "clothes": bool(ss.get("garments")),
        "plan": ss.get("plan") is not None,
        "sharing": ss.get("sharing_seen", False),
    }.get(step, False)


def workers() -> int:
    return min(8, os.cpu_count() or 1)


# ---- save / load ------------------------------------------------------------------------------
def to_json() -> str:
    ss = st.session_state
    gs = []
    for g in ss.garments:
        d = dataclasses.asdict(g)
        gs.append(d)
    return json.dumps({"version": 1, "people": ss.people, "preset": ss.preset,
                       "compartments": [dataclasses.asdict(c) for c in ss.wardrobe.compartments], "garments": gs}, indent=1)


def from_json(text: str) -> None:
    d = json.loads(text)
    ss = st.session_state
    ss.people = d["people"]
    ss.preset = d.get("preset", "two_door")
    w = load_preset(ss.preset)
    if d.get("compartments"):
        w = dataclasses.replace(w, compartments=[Compartment(**c) for c in d["compartments"]])
    ss.wardrobe = w
    out = []
    for g in d["garments"]:
        if g.get("outline"):
            g["outline"] = tuple(tuple(p) for p in g["outline"])
        out.append(Garment(**g))
    ss.garments = out
    ss.mode = "own"
    reset_plans()
