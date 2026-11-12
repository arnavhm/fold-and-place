from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from fold_and_place.app import fold3d, state, ui
from fold_and_place.storage.baselines import BASELINES
from fold_and_place.storage.checker import check
from fold_and_place.storage.model import SolveConfig, solve

ui.setup("Your plan", "plan")
w, gs = state.wardrobe(), state.garments()
if not gs:
    st.info("Add some clothes first.")
    st.page_link("views/clothes.py", label="Go to your clothes", icon=":material/arrow_forward:")
    st.stop()

MODE_WORDS = {"flat": "fold and stack", "file": "fold and stand on its edge", "roll": "roll", "hang": "hang"}
NAMES = {"opt": "This plan", "B1": "The usual way (flat stacks)", "B2": "KonMari filing", "B3": "Hang everything", "B4": "Quick greedy rule"}


def plan_key() -> str:
    blob = json.dumps([[g.id, g.category, g.owner, g.value, g.wear_freq, g.flat_w, g.flat_l, g.outline is not None] for g in gs]
                      + [[c.id, c.kind, c.x, c.z, c.w, c.h, c.d] for c in w.compartments]
                      + [[k, p["height_cm"]] for k, p in st.session_state.people.items()]
                      + [st.session_state.get("lams", (1.0, 1.0))])
    return hashlib.md5(blob.encode()).hexdigest()


key = plan_key()
if st.session_state.get("plan_key") != key:
    lam_r, lam_c = st.session_state.get("lams", (1.0, 1.0))
    with st.spinner(f"Working out the best fold and place for {len(gs)} clothes..."):
        plans = {k: (f(gs, w, lam_reach=lam_r, lam_crease=lam_c) if k == "B4" else f(gs, w)) for k, f in BASELINES.items()}
        plans["opt"] = solve(gs, w, cfg=SolveConfig(time_limit_s=30, workers=state.workers(), lam_reach=lam_r, lam_crease=lam_c))
    st.session_state.plans = plans
    st.session_state.plan = plans["opt"]
    st.session_state.plan_key = key

plans = st.session_state.plans
opt, base = plans["opt"], plans["B1"]
lam_r, lam_c = st.session_state.get("lams", (1.0, 1.0))
m, b = opt.metrics(lam_r, lam_c), base.metrics(lam_r, lam_c)
from fold_and_place.storage.wardrobe import golden_for  # noqa: E402

easy = sum(1 for g in opt.stored if opt.compartment_of(g).zone_cost(golden_for(opt.garments[g].owner)) == 1.0 and opt.garments[g].wear_freq >= 2)
often = sum(1 for g in opt.garments.values() if g.wear_freq >= 2)

head = "Everything fits." if not opt.overflow else f"{len(opt.overflow)} clothes don't fit - the ones that matter least."
head += f" {easy} of the {often} clothes worn every week or two are within easy reach." if often else ""
ui.raw(f'<div class="fp-lead">{head}</div>')
effort_cut = 100 * (1 - m["effort_incl_overflow"] / max(b["effort_incl_overflow"], 1e-9))
crease_cut = 100 * (1 - m["crease_load"] / max(b["crease_load"], 1e-9))
ui.big_numbers([
    (f"{m['stored']} / {len(gs)}", "clothes stored" + (f" (+{m['stored'] - b['stored']} vs the usual way)" if m["stored"] > b["stored"] else ""), "good" if not opt.overflow else ""),
    (f"{effort_cut:.0f}% less", "reaching and bending than the usual way", "good" if effort_cut > 0 else ""),
    (f"{crease_cut:.0f}% fewer", "creases on delicate fabrics", "good" if crease_cut > 0 else ""),
    ("Best possible ✓" if (m["gap"] is not None and m["gap"] < 1e-4) else f"within {100 * (m['gap'] or 0):.1f}% of best", "proven by the solver", ""),
])
st.caption("“The usual way” = every garment folded flat in stacks, shelves filled top to bottom. "
           "Reaching and bending counts how often you reach outside your easy-reach band, weighted by how often you wear each thing.")

view = st.segmented_control("Show", list(NAMES), default="opt", format_func=lambda k: NAMES[k], key="plan_view")
shown = plans[view or "opt"]
left, right = st.columns([5, 6])
comps = sorted(w.compartments, key=lambda c: (-c.z, c.x))
with right:
    labels = {c.id: f"{ui.compartment_name(c, w)} ({sum(len(ln.items) for ln in shown.lanes_in(c.id))})" for c in comps}
    busiest = max(comps, key=lambda c: sum(len(ln.items) for ln in shown.lanes_in(c.id))).id
    sel = st.pills("Look inside", [c.id for c in comps], default=busiest, format_func=lambda i: labels[i], key="plan_comp")
    sel = sel or busiest
    c = w.get(sel)
    items = [gid for lane in shown.lanes_in(sel) for gid in lane.items]
    st.markdown(f"**{ui.compartment_name(c, w)}** · {ui.compartment_reach(c)} · {len(items)} items")
    box = st.container(height=470, border=False)
    with box:
        if not items:
            st.caption("Empty.")
        for gid in items:
            g, o = shown.garments[gid], shown.choice[gid]
            r = st.columns([1, 6, 2])
            ui.raw(where=r[0], html=ui.thumb(g, 44))
            ui.raw(where=r[1], html=f"<b>{g.label}</b> {ui.chip(g.owner)}<br><span class='fp-soft'>{MODE_WORDS[o.mode].capitalize()}: {o.method}. "
                      f"Worn {ui.wear_words(g.wear_freq)}.</span>")
            if o.steps and r[2].button("Watch", key=f"wf_{view}_{gid}", icon=":material/play_circle:", help="Watch how to fold it"):
                st.session_state.fold_gid = gid
                st.session_state.fold_view = view
with left:
    hl = st.session_state.get("fold_gid")
    ui.raw(ui.wardrobe_svg(w, shown, selected=sel, highlight=hl, width=470))
    ui.raw(ui.legend(list(st.session_state.people)) + " <span class='fp-soft'>· hover any item for details</span>")

if shown.overflow:
    with st.expander(f"Doesn't fit ({len(shown.overflow)})"):
        ui.raw(" ".join(f"{ui.chip(shown.garments[g].owner)} {shown.garments[g].label}" for g in shown.overflow))


@st.dialog("How to fold it", width="large")
def fold_dialog(gid: str, view: str):
    p = st.session_state.plans[view]
    g, o = p.garments[gid], p.choice[gid]
    st.markdown(f"**{g.label}** ({ui.person_label(g.owner)}) - {o.method}")
    page = fold3d.html(g, o.steps, colour=ui.person_colour(g.owner), height=440, title=g.label,
                       done=f"Done - {MODE_WORDS[o.mode]}: about {o.a:.0f} × {o.b:.0f} × {o.h:.1f} cm.")
    st.iframe(page, height=450)
    for k, s in enumerate(o.steps, 1):
        st.markdown(f"{k}. {s.text}")


if st.session_state.get("fold_gid") and st.session_state.get("fold_view") == view:
    gid = st.session_state.pop("fold_gid")
    if gid in shown.garments and shown.choice.get(gid) is not None:
        fold_dialog(gid, view)

st.subheader("Per person")
cols = st.columns(max(1, len(st.session_state.people)))
for col, (k, _) in zip(cols, st.session_state.people.items()):
    mine = [g for g in opt.garments.values() if g.owner == k]
    if not mine:
        continue
    stored = [g for g in mine if g.id in opt.stored]
    reach = sum(1 for g in stored if opt.compartment_of(g.id).zone_cost(golden_for(k)) == 1.0)
    ui.raw(where=col, html=f"<div class='fp-card'>{ui.chip(k)}<br><b>{len(stored)} of {len(mine)}</b> stored<br>"
             f"<span class='fp-soft'>{reach} within easy reach</span></div>")

with st.expander("Show the maths"):
    rows = [p.metrics(lam_r, lam_c) | {"plan": NAMES[k]} for k, p in plans.items()]
    df = pd.DataFrame(rows).set_index("plan")
    st.dataframe(df[["stored", "overflow", "stored_value", "effort_incl_overflow", "crease_load", "packet_fill_pct", "rod_fill_pct", "objective", "solve_s", "gap"]]
                 .rename(columns={"effort_incl_overflow": "reach effort", "crease_load": "crease load", "packet_fill_pct": "shelf fill %", "rod_fill_pct": "rail fill %"}),
                 width="stretch")
    st.markdown(
        "The plan comes from an exact constraint-programming model (OR-Tools CP-SAT). It picks one fold for every garment "
        "and a place for it - stacks on shelves, rows standing on edge in drawers, hangers on rails - so that everything "
        "fits the real dimensions. It maximises **20 × value stored − reach effort − crease load**: value first, then "
        "easy reach for what you wear often, then fewer creases on delicate fabrics. *Gap* is how far from the proven best."
    )
    issues = check(opt)
    st.caption("Independent check: " + ("every item fits where it was put ✓" if not issues else "; ".join(map(str, issues[:3]))))
    a, b2 = st.columns(2)
    nr = a.slider("Weight on easy reach", 0.0, 3.0, lam_r, 0.25)
    nc = b2.slider("Weight on fewer creases", 0.0, 3.0, lam_c, 0.25)
    if (nr, nc) != (lam_r, lam_c) and st.button("Re-plan with these weights"):
        st.session_state.lams = (nr, nc)
        st.rerun()

d1, d2 = st.columns(2)
from fold_and_place.app.printable import plan_html  # noqa: E402

d1.download_button("Printable plan (shelf by shelf)", plan_html(opt), file_name="wardrobe_plan.html", mime="text/html",
                   icon=":material/print:", width="stretch")
d2.download_button("Plan as a spreadsheet (CSV)", pd.DataFrame(opt.rows()).to_csv(index=False), file_name="wardrobe_plan.csv",
                   icon=":material/table:", width="stretch")
ui.next_button("Next: sharing fairly →", "views/sharing.py")
