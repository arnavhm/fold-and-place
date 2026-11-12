from __future__ import annotations

import dataclasses

import pandas as pd
import streamlit as st

from fold_and_place import config
from fold_and_place.app import state, ui
from fold_and_place.storage import load_preset

ui.setup("Your wardrobe", "wardrobe", "Pick the almirah your clothes go into, and tell us who uses it. "
         "Each person's height decides which shelves are easy for them to reach.")
st.session_state.wardrobe_seen = True

names = state.presets()
prices = config.load("presets").get("prices_inr", {})
st.subheader("Which wardrobe?")
cols = st.columns(len(names))
for col, (key, label) in zip(cols, names.items()):
    w = load_preset(key)
    on = key == st.session_state.preset
    with col, st.container(border=True):
        ui.raw(ui.wardrobe_svg(w, width=170))
        short = label.split("(")[0].strip()
        st.markdown(f"**{short}**" + (" ✓" if on else ""))
        st.caption(f"{w.width:.0f} × {w.height:.0f} cm" + (f" · ~₹{prices[key]:,}" if key in prices else ""))
        if not on and st.button("Choose", key=f"pick_{key}", width="stretch"):
            state.set_preset(key)
            st.rerun()

st.divider()
left, right = st.columns([5, 4])
w = state.wardrobe()
with left:
    st.subheader(w.label.split("(")[0].strip())
    ui.raw(ui.wardrobe_svg(w, width=440))
with right:
    st.subheader("What's inside")
    rows = []
    for c in sorted(w.compartments, key=lambda c: (-c.z, c.x)):
        cap = f"{c.w:.0f} cm of rail" if c.kind.startswith("rail") else f"{c.w:.0f} × {c.d:.0f} cm, {c.h:.0f} cm tall"
        rows.append({"Space": ui.compartment_name(c, w), "Size": cap, "Reach": ui.compartment_reach(c)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("The yellow bar on the drawing is the easy-reach band: from about knee height up to shoulder height. "
               "Things you wear often belong there.")

    st.subheader("Who uses it?")
    ppl = st.session_state.people
    df = pd.DataFrame([{"key": k, "Name": p["label"], "Height (cm)": int(p["height_cm"])} for k, p in ppl.items()])
    ed = st.data_editor(df, hide_index=True, width="stretch", num_rows="dynamic", disabled=["key"],
                        column_config={"key": None, "Height (cm)": st.column_config.NumberColumn(min_value=90, max_value=210, step=1)},
                        key=f"people_{len(ppl)}")
    if st.button("Save people"):
        new = {}
        for i, r in enumerate(ed.to_dict("records")):
            if not r.get("Name"):
                continue
            k = r.get("key") if isinstance(r.get("key"), str) and r.get("key") else f"p{i}_{str(r['Name']).lower().replace(' ', '_')}"
            new[k] = {"label": str(r["Name"]), "height_cm": float(r.get("Height (cm)") or 165)}
        if new:
            st.session_state.people = new
            state.apply_people()
            state.reset_plans()
            st.success("Saved. Easy-reach shelves now follow these heights.")

with st.expander("Change the shelves yourself"):
    st.caption("Positions in cm: x from the left, z from the floor; w, h, d are width, height and depth.")
    df = pd.DataFrame([dataclasses.asdict(c) for c in w.compartments])
    ed = st.data_editor(df, num_rows="dynamic", width="stretch", key=f"comp_{w.name}",
                        column_config={"kind": st.column_config.SelectboxColumn(options=["shelf", "loft", "drawer", "rail_long", "rail_short"])})
    if st.button("Apply layout"):
        rows = ed.dropna().to_dict("records")
        bad = [r["id"] for r in rows if r["x"] + r["w"] > w.width + 1e-6 or r["z"] + r["h"] > w.height + 1e-6 or r["d"] > w.depth + 1e-6]
        if bad:
            st.error(f"These stick out of the wardrobe: {', '.join(map(str, bad))}")
        else:
            state.set_compartments(rows)
            st.rerun()

ui.next_button("Next: your clothes →", "views/clothes.py")
