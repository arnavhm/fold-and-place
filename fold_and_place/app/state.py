"""Session state shared by the dashboard pages (wardrobe, garments, latest plan)."""

from __future__ import annotations

import dataclasses

import streamlit as st

from fold_and_place import config
from fold_and_place.garments import household_garments
from fold_and_place.garments.record import Garment
from fold_and_place.storage import load_preset
from fold_and_place.storage.wardrobe import Compartment, Wardrobe

OWNER_COLOURS = {"student": "#8fa9c9", "mother": "#d9a75f", "father": "#9cc39c"}


def init() -> None:
    ss = st.session_state
    if "preset" not in ss:
        ss.preset = "two_door"
        ss.wardrobe = load_preset("two_door")
    if "garments" not in ss:
        ss.garments = household_garments()
    ss.setdefault("plan", None)
    ss.setdefault("plans", {})


def wardrobe() -> Wardrobe:
    init()
    return st.session_state.wardrobe


def garments() -> list[Garment]:
    init()
    return st.session_state.garments


def set_preset(name: str) -> None:
    st.session_state.preset = name
    st.session_state.wardrobe = load_preset(name)
    st.session_state.plan = None
    st.session_state.plans = {}


def set_compartments(rows: list[dict]) -> None:
    w = st.session_state.wardrobe
    comps = [Compartment(**{k: (float(v) if k in "xzwhd" else str(v)) for k, v in r.items()}) for r in rows]
    st.session_state.wardrobe = dataclasses.replace(w, compartments=comps, name=w.name + "*")
    st.session_state.plan = None


def owner_colour(owner: str) -> str:
    return OWNER_COLOURS.get(owner, "#b0b0b0")


def presets() -> dict[str, str]:
    return {k: v["label"] for k, v in config.presets().items()}
