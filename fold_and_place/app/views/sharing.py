from __future__ import annotations

import pandas as pd
import streamlit as st

from fold_and_place import config
from fold_and_place.app import ui
from fold_and_place.mechanisms.fairness import fairness, proportional_shares
from fold_and_place.mechanisms.mechanisms import ALL, BY_NAME
from fold_and_place.mechanisms.scenario import Scenario, quantize
from fold_and_place.mechanisms.strategic import collusion_test

ui.setup("Sharing fairly", "sharing", "When not everything fits, someone's clothes have to go. Each person knows how much "
         "their own clothes matter to them - and could exaggerate. Which rule picks well, and which one rewards lying?")
st.session_state.sharing_seen = True

CORE = {
    "Equal split": ("Equal split", "Everyone gets the same share of every shelf, rail and drawer."),
    "Priority by claimed value": ("Loudest claim wins", "Space goes first to whatever people <i>say</i> matters most."),
    "Random serial dictatorship": ("Take turns", "Draw lots for an order; each person in turn takes the best space for their own clothes."),
    "VCG": ("Pay for what you push out", "Space goes where it's valued most; you pay chore credits equal to the value your clothes push out of the wardrobe."),
}
SHORT = {m.name: s for m, s in zip(ALL, ["VCG", "VCG + Bailey", "Myerson auction", "Take turns (RSD)", "Max Nash welfare", "Equal split",
                                         "First come, first served", "Proportional to claims", "Loudest claim (Priority)"])}


@st.cache_resource(show_spinner="Setting up the shared section...")
def scenario() -> Scenario:
    return Scenario.default()


@st.cache_resource(show_spinner=False, max_entries=256)
def run(mech: str, liar: str | None, mult: float):
    sc = scenario()
    true = sc.true_values()
    rep = dict(true)
    if liar:
        for g in sc.owned(liar):
            rep[g.id] = quantize(true[g.id] * mult)
    return BY_NAME[mech].run(sc, rep)


@st.cache_resource(show_spinner=False)
def best_welfare() -> float:
    sc = scenario()
    return sc.welfare(sc.optimum(sc.true_values()), sc.true_values())


@st.cache_resource(show_spinner=False)
def shares():
    sc = scenario()
    return proportional_shares(sc, sc.true_values())


sc = scenario()
true = sc.true_values()
agents = sc.agents
label = lambda a: config.personas()[a]["label"]  # noqa: E731
colour = {a: ui.PALETTE[i % len(ui.PALETTE)] for i, a in enumerate(agents)}

left, right = st.columns([2, 3])
with left:
    ui.raw(ui.wardrobe_svg(sc.wardrobe, width=260))
with right:
    counts = {a: len(sc.owned(a)) for a in agents}
    st.markdown(f"**The set-up.** {', '.join(f'{label(a)} ({counts[a]} clothes)' for a in agents)} share one small section: "
                f"a shelf, a short rail and a drawer. **{len(sc.garments)} clothes, but only about {len(sc.optimum(true).stored)} fit.**")
    st.markdown("Each garment has a value to its owner from 1 to 10 that only they know. They tell the rule a number - "
                "honestly or not. We then judge every rule with the **true** values.")
    with st.expander("Why “chore credits”?"):
        st.markdown("One rule asks people to pay. In a family nobody pays rupees for shelf space, so payments are "
                    "**chore credits**: 1 credit = 1 point of value, settled in chores (one load of laundry, say). "
                    "You pay exactly the value your clothes push out of the wardrobe for everyone else.")

st.subheader("Does anyone exaggerate?")
c1, c2 = st.columns([3, 2])
liar = c1.segmented_control("Who exaggerates", ["Nobody"] + agents, default="Nobody", format_func=lambda a: a if a == "Nobody" else label(a), key="liar")
liar = None if liar in (None, "Nobody") else liar
mult = 1.0
if liar:
    mult = c2.segmented_control("Claims their clothes matter", [0.5, 1.5, 2.0, 3.0], default=3.0, format_func=lambda m: f"×{m:g}", key="mult") or 3.0

best = best_welfare()
ps = shares()
cards = st.columns(4)
verdicts = {}
for col, (name, (title, how)) in zip(cards, CORE.items()):
    with st.spinner(f"{title}..."):
        out = run(name, liar, mult)
        honest = run(name, None, 1.0)
    eff = out.welfare(sc, true) / best
    fair = fairness(sc, out, true, ps)
    tags = [f'<span class="fp-tag {"ok" if eff > 0.97 else ("mid" if eff > 0.8 else "no")}">uses {eff:.0%} of the wardrobe\'s value</span>',
            f'<span class="fp-tag {"ok" if fair["min_share_ratio"] >= 0.999 else "mid"}">'
            + ("nobody worse off than an equal split" if fair["min_share_ratio"] >= 0.999 else f"someone gets {fair['min_share_ratio']:.0%} of an equal split")
            + "</span>"]
    if liar:
        gain = out.utility(sc, liar, true) - honest.utility(sc, liar, true)
        verdicts[name] = gain
        tags.append(f'<span class="fp-tag {"no" if gain > 0.05 else "ok"}">'
                    + (f"lying paid off: +{gain:.1f}" if gain > 0.05 else (f"lying backfired: {gain:.1f}" if gain < -0.05 else "lying changed nothing")) + "</span>")
    pays = ""
    if out.payments:
        pays = "<div class='fp-soft' style='margin-top:.4rem'>Chore credits: " + ", ".join(
            f"{label(a)} {out.payments.get(a, 0) - out.rebates.get(a, 0):.1f}" for a in agents) + "</div>"
    stored = " · ".join(f"{label(a)} {fair['stored_value'][a]:.0f}" for a in agents)
    ui.raw(where=col, html=f'<div class="fp-verdict"><h4>{title}</h4><div class="how">{how}</div>{"".join(tags)}'
             f"<div class='fp-soft' style='margin-top:.4rem'>Value kept: {stored}</div>{pays}</div>")

if liar:
    won = [CORE[n][0] for n, g in verdicts.items() if g > 0.05]
    safe = [CORE[n][0] for n, g in verdicts.items() if g <= 0.05]
    msg = f"**{label(liar)}** says their clothes matter ×{mult:g}. "
    msg += (f"Under **{', '.join(won)}**, that pays off - so everyone would start exaggerating. " if won else "No rule rewards it here. ")
    msg += f"Under **{', '.join(safe)}**, honesty is the best policy."
    st.info(msg, icon=":material/lightbulb:")
else:
    st.info("Everyone is honest. Now pick someone above and make them exaggerate.", icon=":material/lightbulb:")

st.subheader("What does each rule give?")
pick = st.segmented_control("Show the wardrobe under", list(CORE), default="VCG", format_func=lambda n: CORE[n][0], key="show_rule") or "VCG"
out = run(pick, liar, mult)
p = out.plan()
l2, r2 = st.columns([2, 3])
if p is not None:
    saved = dict(st.session_state.get("people", {}))
    st.session_state.people = {a: {"label": label(a), "height_cm": 170} for a in agents}  # colours for the drawing
    try:
        ui.raw(where=l2, html=ui.wardrobe_svg(sc.wardrobe, p, width=280))
        rows = pd.DataFrame(p.rows())
    finally:
        st.session_state.people = saved
    tab = rows.assign(kept=rows.compartment != "OVERFLOW").groupby("owner").agg(kept=("kept", "sum"), clothes=("garment", "count"))
    tab.index = [label(a) for a in tab.index]
    r2.dataframe(tab, width="stretch")
    r2.caption("Hover the drawing to see whose clothes went where.")
else:
    r2.caption("This rule splits the space into personal sections, so there is no single shared picture.")

st.subheader("What if two people team up?")
st.caption("A rule can be safe against one person lying yet open to two people coordinating. We search joint lies "
           "(each scales all their values by ×0.5 to ×3) and report the best combined gain for the pair.")
pairs = [(a, b) for i, a in enumerate(agents) for b in agents[i + 1 :]]
pc1, pc2 = st.columns([2, 3])
pair = pc1.selectbox("Pair", pairs, format_func=lambda t: f"{label(t[0])} + {label(t[1])}")
if pc1.button("Search joint lies (about 30 s)"):
    with st.spinner("Trying every joint lie..."):
        res = [collusion_test(BY_NAME[n], sc, pair) for n in ("VCG", "Random serial dictatorship")]
    for r in res:
        if r["best_gain"] > 0.05:
            pc2.warning(f"**{SHORT[r['mechanism']]}**: together they gain **{r['best_gain']:.1f}** by claiming ×{r['multipliers'][0]:g} and ×{r['multipliers'][1]:g}.")
        else:
            pc2.success(f"**{SHORT[r['mechanism']]}**: no joint lie helps this pair.")
    pc2.caption("VCG is truthful for one person at a time, not for groups: understating together cuts what each pays for the other.")

with st.expander("Show the maths: all nine rules"):
    if st.button("Run all nine with the current exaggeration"):
        rows = []
        for m in ALL:
            o = run(m.name, liar, mult)
            f = fairness(sc, o, true, ps)
            row = {"rule": SHORT[m.name], "money": "yes" if m.uses_money else "no", "efficiency %": round(100 * o.welfare(sc, true) / best, 1)}
            for a in agents:
                row[f"utility {label(a)}"] = round(o.utility(sc, a, true), 2)
            if liar:
                row["lie gain"] = round(o.utility(sc, liar, true) - run(m.name, None, 1.0).utility(sc, liar, true), 2)
            row["min share ratio"] = round(f["min_share_ratio"], 2)
            row["max envy"] = round(f["max_envy"], 1)
            row["what theory says"] = m.claim
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(
        "**Model.** Each person *i* has private values θ for their own garments. A rule maps reports to a plan *x* "
        "(and payments *p*). Utility = value of your stored garments (+ small reach/crease terms) − *p*. "
        "**VCG** picks the plan with the highest reported total and charges each person the drop in everyone else's "
        "best total caused by their presence - so reporting the truth is a dominant strategy. **Take turns** (random serial "
        "dictatorship) is strategy-proof without money. **Loudest claim** and **Proportional** reward inflating claims."
    )
    try:
        pm = pd.read_csv(config.RESULTS_DIR / "E5_property_matrix.csv")
        st.caption("Experiment E5 (hundreds of random lies per rule):")
        st.dataframe(pm, hide_index=True, width="stretch")
    except FileNotFoundError:
        pass
