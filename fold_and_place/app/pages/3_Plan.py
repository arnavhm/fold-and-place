from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from fold_and_place.app import draw, state
from fold_and_place.storage.baselines import BASELINES
from fold_and_place.storage.checker import check
from fold_and_place.storage.model import SolveConfig, solve

st.set_page_config(page_title="Plan · Fold & Place", page_icon="📐", layout="wide")
state.init()
st.title("Plan: fold and place everything")

w, gs = state.wardrobe(), state.garments()
st.caption(f"{len(gs)} garments into **{w.label}**.")

with st.sidebar:
    st.subheader("Optimizer")
    tl = st.slider("Time limit (s)", 5, 120, 30, 5)
    lam_r = st.slider("Weight on reach effort", 0.0, 3.0, 1.0, 0.25, help="Effort per wear: golden zone 1, just outside 1.5, far 3; flat stacks x1.3.")
    lam_c = st.slider("Weight on creasing", 0.0, 3.0, 1.0, 0.25, help="Crease risk of the fabric x number of folds.")
    st.caption("Stored value always dominates: one value point = 20 effort points.")

if st.button("Solve and compare with baselines", type="primary"):
    plans = {}
    with st.spinner("Rule-of-thumb baselines..."):
        for k, f in BASELINES.items():
            p = f(gs, w, lam_reach=lam_r, lam_crease=lam_c) if k == "B4" else f(gs, w)
            plans[p.method] = p
    with st.spinner(f"CP-SAT optimizer (up to {tl} s)..."):
        cfg = SolveConfig(time_limit_s=tl, workers=min(8, os.cpu_count() or 1), lam_reach=lam_r, lam_crease=lam_c)
        opt = solve(gs, w, cfg=cfg)
    plans[opt.method] = opt
    st.session_state.plans = plans
    st.session_state.plan = opt
    st.session_state.lams = (lam_r, lam_c)

plans = st.session_state.plans
plan = st.session_state.plan
if not plans:
    st.info("Press the button to run the optimizer and four baselines (traditional flat stacks, KonMari filing, hang-first, greedy).")
    st.pyplot(draw.wardrobe_front(w), width="content")
    st.stop()

lam_r, lam_c = st.session_state.get("lams", (1.0, 1.0))
rows = [p.metrics(lam_r, lam_c) for p in plans.values()]
cmp = pd.DataFrame(rows).set_index("method")
cols = ["stored", "overflow", "stored_value", "retrieval_effort", "effort_incl_overflow", "crease_load", "packet_fill_pct", "rod_fill_pct", "objective", "solve_s", "gap"]
st.subheader("Comparison")
st.dataframe(
    cmp[cols].style.format(precision=1, na_rep="-").highlight_max(subset=["stored", "stored_value", "objective"], color="#d8eed8").highlight_min(
        subset=["overflow", "effort_incl_overflow", "crease_load"], color="#d8eed8"
    ),
    width="stretch",
)
best = cmp.loc[plan.method]
b1 = cmp.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stored", f"{int(best.stored)} / {len(gs)}", f"{int(best.stored - b1.stored):+d} vs flat stacks")
c2.metric("Effort (incl. chair pile)", f"{best.effort_incl_overflow:.0f}", f"{100 * (best.effort_incl_overflow / b1.effort_incl_overflow - 1):+.0f}%", delta_color="inverse")
c3.metric("Crease load", f"{best.crease_load:.0f}", f"{100 * (best.crease_load / max(b1.crease_load, 1e-9) - 1):+.0f}%", delta_color="inverse")
c4.metric("Optimality gap", "0 (optimal)" if (best.gap is not None and best.gap < 1e-4) else (f"{best.gap:.2%}" if best.gap is not None else "-"))

issues = check(plan)
if issues:
    st.error("Checker found problems: " + "; ".join(map(str, issues[:5])))
else:
    st.caption("Independent checker: every stack, row and rail fits its compartment; every garment is either placed once or in the overflow pile.")

st.divider()
pick = st.selectbox("Show plan", list(plans), index=list(plans).index(plan.method))
shown = plans[pick]
tbl = pd.DataFrame(shown.rows())
left, right = st.columns([2, 3])
with right:
    owners = sorted(tbl.owner.unique())
    who = st.multiselect("Owner", owners, default=owners, key="plan_owner")
    sub = tbl[tbl.owner.isin(who)].sort_values(["compartment", "position"])
    sel = st.dataframe(sub, hide_index=True, width="stretch", height=470, on_select="rerun", selection_mode="single-row")
    hl = None
    if sel and sel.selection.rows:
        hl = sub.iloc[sel.selection.rows[0]].garment
        st.session_state.fold_gid = hl
        st.caption(f"Selected **{hl}** - open the Fold studio to watch its fold.")
with left:
    st.pyplot(draw.wardrobe_front(w, shown, highlight=hl), width="stretch")
    st.caption("Colours: student blue, mother amber, father green. Drawers are seen from the front: back rows are paler.")

st.download_button("Download plan (CSV)", tbl.to_csv(index=False), file_name=f"plan_{w.name}.csv")
