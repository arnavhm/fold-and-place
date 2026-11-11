from __future__ import annotations

import pandas as pd
import streamlit as st

from fold_and_place import config
from fold_and_place.app import draw
from fold_and_place.mechanisms.fairness import fairness, proportional_shares
from fold_and_place.mechanisms.mechanisms import ALL
from fold_and_place.mechanisms.scenario import Scenario, quantize

st.set_page_config(page_title="Mechanism lab · Fold & Place", page_icon="⚖️", layout="wide")
st.title("Mechanism lab: who gets the space?")
st.markdown(
    "Three people share one small wardrobe and not everything fits. Each person knows how much their own clothes "
    "matter to them (a **private value**, 1-10 per garment) and **reports** a number. A mechanism turns the reports "
    "into a plan (and, for some, payments in *space credits*). Below, everyone's utility is measured with their "
    "**true** values - so you can see who gains by lying."
)


@st.cache_resource(show_spinner="Setting up the shared-wardrobe scenario...")
def scenario() -> Scenario:
    return Scenario.default()


sc = scenario()
true = sc.true_values()
agents = sc.agents
SHORT = {m.name: s for m, s in zip(ALL, ["VCG", "VCG+Bailey", "Myerson", "RSD", "MNW", "Equal split", "FCFS", "Proportional", "Priority"])}

with st.expander(f"The scenario: {len(sc.garments)} garments, {sc.wardrobe.label}", expanded=False):
    c1, c2 = st.columns([2, 3])
    c1.pyplot(draw.wardrobe_front(sc.wardrobe), width="stretch")
    c2.dataframe(
        pd.DataFrame([{"owner": g.owner, "garment": g.id, "category": g.label, "true value": g.value, "wears/month": g.wear_freq} for g in sc.garments]),
        hide_index=True, width="stretch", height=380,
    )

st.subheader("1 · What does each person report?")
cols = st.columns(len(agents))
mult = {}
for a, c in zip(agents, cols):
    mult[a] = c.select_slider(f"{config.personas()[a]['label']} claims x true value", options=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0], value=1.0, key=f"m_{a}")
with st.expander("...or edit individual garments' reports"):
    base = pd.DataFrame([{"garment": g.id, "owner": g.owner, "true": g.value, "reported": quantize(g.value * mult[g.owner])} for g in sc.garments])
    ed = st.data_editor(base, disabled=["garment", "owner", "true"], hide_index=True, width="stretch", height=300,
                        column_config={"reported": st.column_config.NumberColumn(min_value=0.0, max_value=30.0, step=0.5)}, key=f"rep_{'_'.join(map(str, mult.values()))}")
reports = {r["garment"]: float(r["reported"]) for r in ed.to_dict("records")}
liars = [a for a in agents if any(abs(reports[g.id] - true[g.id]) > 1e-9 for g in sc.owned(a))]

pick = st.multiselect("Mechanisms", [m.name for m in ALL], default=[m.name for m in ALL], format_func=lambda n: SHORT[n])

if st.button("Run", type="primary"):
    rows, outs = [], {}
    opt = sc.welfare(sc.optimum(true), true)
    ps = proportional_shares(sc, true)
    prog = st.progress(0.0)
    sel = [m for m in ALL if m.name in pick]
    for k, m in enumerate(sel):
        out = m.run(sc, reports)
        honest = m.run(sc, true) if liars else out
        outs[m.name] = out
        fair = fairness(sc, out, true, ps)
        row = {
            "mechanism": SHORT[m.name],
            "money": "yes" if m.uses_money else "no",
            "welfare (true)": round(out.welfare(sc, true), 2),
            "efficiency %": round(100 * out.welfare(sc, true) / opt, 1),
        }
        for a in agents:
            row[f"u {a}"] = round(out.utility(sc, a, true), 2)
        for a in agents:
            row[f"pays {a}"] = round(out.payments.get(a, 0.0) - out.rebates.get(a, 0.0), 2)
        for a in liars:
            row[f"lie gain {a}"] = round(out.utility(sc, a, true) - honest.utility(sc, a, true), 2)
        row["min share ratio"] = round(fair["min_share_ratio"], 2)
        row["max envy"] = round(fair["max_envy"], 1)
        row["claim"] = m.claim
        rows.append(row)
        prog.progress((k + 1) / len(sel))
    st.session_state.lab = (pd.DataFrame(rows), outs, liars, reports)

if "lab" in st.session_state:
    df, outs, liars_run, _ = st.session_state.lab
    st.subheader("2 · Outcomes")
    gain_cols = [c for c in df.columns if c.startswith("lie gain")]
    sty = df.style.format(precision=2)
    if gain_cols:
        sty = sty.map(lambda v: "background-color:#f6d5d5" if v > 1e-6 else ("background-color:#d8eed8" if v < -1e-6 else ""), subset=gain_cols)
    st.dataframe(sty, hide_index=True, width="stretch")
    if liars_run:
        st.markdown(
            f"**{', '.join(liars_run)}** misreported. A red *lie gain* means lying paid off under that mechanism. "
            "VCG, VCG+Bailey and RSD should never show red (dominant-strategy truthful); Proportional and Priority often do. "
            "Myerson is truthful **per garment** (one value at a time), not when a person shades several values at once."
        )
    else:
        st.caption("Everyone reported truthfully. Move a slider above to make someone lie, then run again.")
    st.markdown(
        "*Utility* = value of stored garments (+ small reach/crease terms) minus credits paid. "
        "*Efficiency* = true welfare / best possible. *Min share ratio* < 1 means someone got less than an equal split "
        "would give them. *Max envy*: value points someone would gain by swapping for another's space."
    )
    show = st.selectbox("Show the plan of", list(outs), format_func=lambda n: SHORT[n])
    out = outs[show]
    p = out.plan()
    if p is None:
        prob, plans = max(out.lottery, key=lambda t: t[0])
        st.caption(f"{SHORT[show]} gives each person their own share (or a lottery over {len(out.lottery)} orders); showing the first.")
        c = st.columns(len(agents))
        for a, col in zip(agents, c):
            col.markdown(f"**{a}**")
            col.pyplot(draw.wardrobe_front(plans[a].wardrobe, plans[a]), width="stretch")
    else:
        c1, c2 = st.columns([2, 3])
        c1.pyplot(draw.wardrobe_front(sc.wardrobe, p), width="stretch")
        rows = pd.DataFrame(p.rows())
        summ = rows.assign(stored=rows.compartment != "OVERFLOW").groupby("owner").agg(
            stored=("stored", "sum"), garments=("garment", "count"))
        c2.markdown("**Stored per person**")
        c2.dataframe(summ, width="stretch")
        c2.dataframe(rows[["garment", "owner", "value", "fold", "compartment"]], hide_index=True, width="stretch", height=320)
    if show.startswith("VCG"):
        st.caption("VCG payment = the harm you do to the others: their best welfare without you minus their welfare in the chosen plan. "
                   "When several plans are optimal the solver may pick a different one on each run; payments can then shift "
                   "between people, but each person's utility stays the same (it equals total welfare minus the others' best without you).")

st.divider()
st.subheader("Theory vs experiment (E5)")
try:
    pm = pd.read_csv(config.RESULTS_DIR / "E5_property_matrix.csv")
    st.dataframe(pm, hide_index=True, width="stretch")
    st.caption("From `make experiments` (results/E5_*). Profitable-lie rate = share of random misreports that raised the liar's true utility.")
except FileNotFoundError:
    st.info("Run `make experiments` to fill this table.")
