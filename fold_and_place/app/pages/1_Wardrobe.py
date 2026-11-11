from __future__ import annotations

import dataclasses

import pandas as pd
import streamlit as st

from fold_and_place import config
from fold_and_place.app import draw, state

st.set_page_config(page_title="Wardrobe · Fold & Place", page_icon="🚪", layout="wide")
state.init()
st.title("Wardrobe")

names = state.presets()
keys = list(names)
cur = st.session_state.preset if st.session_state.preset in names else keys[0]
pick = st.selectbox("Preset", keys, index=keys.index(cur), format_func=lambda k: names[k])
if pick != st.session_state.preset:
    state.set_preset(pick)
w = state.wardrobe()
price = config.load("presets").get("prices_inr", {}).get(pick)

left, right = st.columns([2, 3])
with left:
    st.pyplot(draw.wardrobe_front(w, st.session_state.plan), width="stretch")
with right:
    lo, hi = config.golden_zone()
    st.markdown(
        f"**{w.width:.0f} x {w.height:.0f} x {w.depth:.0f} cm** (w x h x d) · **{len(w.compartments)}** compartments · "
        f"**{w.rod_cm:.0f} cm** of hanging rail · **{w.packet_volume / 1000:.0f} L** of shelf and drawer space"
        + (f" · typical price **₹{price:,}**" if price else "")
    )
    st.markdown(
        f"The yellow band is the **golden zone** ({lo:.0f}-{hi:.0f} cm): reachable without bending or a stool. "
        "Each household member's upper limit is their own shoulder height. Space outside it costs more effort per wear "
        "(x1.5 just outside, x3 far outside or in the loft); flat stacks cost x1.3 because you dig to reach the lower ones."
    )
    st.markdown("**Compartments** - edit and apply to try your own layout (cm, x from the left, z from the floor).")
    df = pd.DataFrame([dataclasses.asdict(c) for c in w.compartments])
    ed = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        column_config={"kind": st.column_config.SelectboxColumn(options=["shelf", "loft", "drawer", "rail_long", "rail_short"])},
        key=f"comp_{w.name}",
    )
    if st.button("Apply layout"):
        rows = ed.dropna().to_dict("records")
        bad = [r["id"] for r in rows if r["x"] + r["w"] > w.width + 1e-6 or r["z"] + r["h"] > w.height + 1e-6 or r["d"] > w.depth + 1e-6]
        if bad:
            st.error(f"Outside the carcass: {', '.join(map(str, bad))}")
        else:
            state.set_compartments(rows)
            st.success("Layout updated. Re-run the plan.")
            st.rerun()
st.caption("Presets follow common Indian modular wardrobes (loft above, long and short hanging, shelves, drawers). "
           "Dimensions and prices are typical catalogue figures, not one brand.")
