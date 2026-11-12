from __future__ import annotations

import pandas as pd
import streamlit as st

from fold_and_place import config
from fold_and_place.app import draw, fold3d, state, ui
from fold_and_place.folding.options import generate_options
from fold_and_place.garments.record import make_garment

ui.setup("Fold any garment", lead="Every sensible way to fold a garment, with the packet each one makes. "
         "Pick one to watch it fold. Garments added from a photo fold their own photographed shape.")

cats = config.categories()
gs = state.garments()
plan = st.session_state.get("plan")
src = st.segmented_control("Garment", ["From your clothes", "Any type"], default="From your clothes" if gs else "Any type")
if src == "From your clothes" and gs:
    ids = [g.id for g in gs]
    gid = st.selectbox("Which one", ids, format_func=lambda i: next(f"{g.label} - {ui.person_label(g.owner)} ({i})" for g in gs if g.id == i))
    g = next(x for x in gs if x.id == gid)
else:
    c1, c2 = st.columns(2)
    cat = c1.selectbox("Type", list(cats), format_func=lambda c: cats[c]["label"])
    size = c2.selectbox("Size", list(config.size_scale()), index=1)
    g = make_garment(cat, f"demo_{cat}", size)

opts = generate_options(g)
chosen = plan.choice.get(g.id) if plan is not None and g.id in plan.garments else None
if chosen is not None:
    opts = sorted(opts, key=lambda o: o.id != chosen.id)
words = {"flat": "stack", "file": "stand on edge", "roll": "roll", "hang": "hang"}
tbl = pd.DataFrame([{
    "": "★ in your plan" if chosen is not None and o.id == chosen.id else "",
    "How": o.method, "Then": words[o.mode],
    "Packet": f"{o.a:.0f} × {o.b:.0f} × {o.h:.1f} cm" if o.mode != "hang" else f"{o.rod_in:g} in of rail",
    "Folds": len(o.steps), "Creases": o.creases,
} for o in opts])
a, b = st.columns([1, 5])
ui.raw(where=a, html=ui.thumb(g, 72))
b.markdown(f"**{g.label}** · flat {g.flat_w:.0f} × {g.flat_l:.0f} cm · {g.fabric.replace('_', ' ')}"
           + (" · heavy embroidery: fold flat only, nothing on top" if g.no_load_on_top else "")
           + (" · **its own photographed shape**" if g.outline else ""))
sel = st.dataframe(tbl, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", height=250)
i = sel.selection.rows[0] if sel and sel.selection.rows else 0
o = opts[i]
st.subheader(o.method)
if not o.steps:
    st.info(f"Just hang it: {o.rod_in:g} in of rail, {o.drop:.0f} cm drop." if o.mode == "hang" else "No folding needed.")
else:
    page = fold3d.html(g, o.steps, colour=ui.person_colour(g.owner) if g.owner else "#D9822B", height=480,
                       title=f"{g.label}: {o.method}", done=f"Done - {words[o.mode]}: about {o.a:.0f} × {o.b:.0f} × {o.h:.1f} cm.")
    st.iframe(page, height=490)
    c1, c2 = st.columns([1, 3])
    c1.download_button("Save this animation", page, file_name=f"fold_{g.category}.html", mime="text/html")
    c2.caption("Drag to turn it, scroll to zoom. The red line is the next fold. The fold geometry is exact; "
               "the layer heights in the animation are a sketch.")
    with st.expander("Step by step (2D)"):
        for k, s in enumerate(o.steps, 1):
            st.markdown(f"{k}. {s.text}")
        st.pyplot(draw.fold_steps_figure(g, o.steps), width="stretch")
