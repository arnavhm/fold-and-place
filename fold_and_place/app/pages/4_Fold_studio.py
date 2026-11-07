from __future__ import annotations

import pandas as pd
import streamlit as st

from fold_and_place import config
from fold_and_place.app import draw, fold3d, state
from fold_and_place.folding.options import generate_options
from fold_and_place.garments.record import make_garment

st.set_page_config(page_title="Fold studio · Fold & Place", page_icon="🧺", layout="wide")
state.init()
st.title("Fold studio")

cats = config.categories()
plan = st.session_state.plan
gs = state.garments()

src = st.radio("Garment", ["From the household", "Any category"], horizontal=True)
if src == "From the household":
    ids = [g.id for g in gs]
    default = st.session_state.get("fold_gid")
    gid = st.selectbox("Garment", ids, index=ids.index(default) if default in ids else 0,
                       format_func=lambda i: f"{i}  ({next(g.label for g in gs if g.id == i)})")
    g = next(x for x in gs if x.id == gid)
else:
    c1, c2 = st.columns(2)
    cat = c1.selectbox("Category", list(cats), format_func=lambda c: f"{cats[c]['label']} ({cats[c]['group']})")
    size = c2.selectbox("Size", list(config.size_scale()), index=1)
    g = make_garment(cat, f"demo_{cat}", size)

opts = generate_options(g)
chosen = plan.choice.get(g.id) if plan is not None and g.id in plan.garments else None
if chosen is not None:  # the plan's fold first
    opts = sorted(opts, key=lambda o: o.id != chosen.id)
tbl = pd.DataFrame(
    [
        {
            "": "★ plan" if chosen is not None and o.id == chosen.id else "",
            "method": o.method,
            "mode": o.mode,
            "packet (cm)": f"{o.a:.0f} x {o.b:.0f} x {o.h:.1f}" if o.mode != "hang" else f"{o.rod_in:g} in of rod, {o.drop:.0f} cm drop",
            "plies": o.plies,
            "creases": o.creases,
            "steps": len(o.steps),
        }
        for o in opts
    ]
)
st.caption(f"**{g.label}** · flat {g.flat_w:.0f} x {g.flat_l:.0f} cm · {g.fabric} · crease risk {g.crease_risk:.2f}"
           + (" · heavy embroidery: flat only, nothing on top" if g.no_load_on_top else "")
           + f" · {len(opts)} options after dropping dominated ones")
start = next((i for i, o in enumerate(opts) if chosen is not None and o.id == chosen.id), 0)
sel = st.dataframe(tbl, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", height=260)
i = sel.selection.rows[0] if sel and sel.selection.rows else start
o = opts[i]

st.subheader(o.method)
if o.mode == "hang":
    st.info(f"Hang it: {o.rod_in:g} in of rail, {o.drop:.0f} cm drop. "
            + ("Fold it over the hanger bar first." if o.steps else "No folding needed."))
if o.steps:
    colour = state.owner_colour(g.owner) if g.owner else "#d9a75f"
    done = {"flat": "Done - lay it flat in a stack: ", "file": "Done - stand it on its edge in a drawer row: ",
            "roll": "Done - now roll it from the hem: ", "hang": "Done - hang it: "}[o.mode] + o.summary().split(" - ", 1)[-1]
    page = fold3d.html(g, o.steps, colour=colour, height=500, title=f"{g.label}: {o.method}", done=done)
    if hasattr(st, "iframe"):
        st.iframe(page, height=510)
    else:  # older Streamlit
        import streamlit.components.v1 as components

        components.html(page, height=510)
    c1, c2 = st.columns([1, 3])
    c1.download_button("Download this animation (HTML)", page, file_name=f"fold_{g.category}_{i}.html", mime="text/html")
    c2.caption("Drag to orbit, scroll to zoom. The red line is the next crease. Geometry is exact (polygon folding); "
               "heights are a sketch (one fabric gap per layer) - the packet thickness above uses fabric thickness and loft.")
    with st.expander("Step-by-step diagram (2D)"):
        for k, s in enumerate(o.steps, 1):
            st.markdown(f"{k}. {s.text}")
        st.pyplot(draw.fold_steps_figure(g, o.steps), width="stretch")
if o.roll:
    st.caption("Roll: after the folds above, roll tightly from the hem; the roll's diameter is the packet height.")
