"""Fold & Place dashboard. Run: make app   (or: streamlit run fold_and_place/app/Home.py)"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fold_and_place import config
from fold_and_place.app import state

st.set_page_config(page_title="Fold & Place", page_icon="🧺", layout="wide")
state.init()

st.title("Fold & Place")
st.markdown(
    "**How should a family fold everything they own, where should each piece go in a shared "
    "wardrobe, and how do you split that wardrobe fairly when everyone thinks their clothes matter most?**"
)

c1, c2, c3, c4 = st.columns(4)
c1.markdown("#### 1 · See\nA photo becomes a garment record: category (31 Indian + Western types), "
            "fabric, and flat size from an A4 sheet in the frame or a size chart.")
c2.markdown("#### 2 · Fold\nEach garment gets every sensible fold - flat stack, KonMari file, roll, hang - "
            "simulated as real geometry, with packet size, plies and creases.")
c3.markdown("#### 3 · Place\nA CP-SAT model picks one fold per garment and packs shelves, drawers and rails, "
            "trading off space, reach (golden zone) and creasing.")
c4.markdown("#### 4 · Share\nPeople report how much they value their clothes. VCG, Myerson, serial dictatorship "
            "and others turn those reports into a shared plan - and we test who can cheat.")

st.divider()
st.subheader("Headline results")
res = config.RESULTS_DIR
m1, m2, m3, m4 = st.columns(4)
try:
    sav = pd.read_csv(res / "E4_savings.csv").set_index("baseline")
    b1 = sav.loc["B1 traditional flat stacks"]
    m1.metric("Reach effort vs flat stacks", f"-{b1.effort_saving_pct:.0f}%", f"+{int(b1.extra_items_stored)} items stored")
    m2.metric("Crease load vs flat stacks", f"-{b1.crease_saving_pct:.0f}%")
except FileNotFoundError:
    m1.info("Run `make experiments` for E4.")
try:
    pm = pd.read_csv(res / "E5_property_matrix.csv").set_index("short")
    m3.metric("VCG: efficiency / profitable lies", f"{pm.loc['VCG', 'efficiency_pct']:.0f}% / {pm.loc['VCG', 'profitable_lie_rate']:.0%}")
    m4.metric("Priority-by-claim: profitable lies", f"{pm.loc['Priority', 'profitable_lie_rate']:.0%}", "manipulable", delta_color="inverse")
except (FileNotFoundError, KeyError):
    m3.info("Run `make experiments` for E5.")

st.divider()
left, right = st.columns([2, 3])
with left:
    st.subheader("Current household")
    gs = state.garments()
    df = pd.DataFrame([{"owner": g.owner, "garment": g.label} for g in gs])
    st.dataframe(df.groupby("owner").size().rename("garments"), width="stretch")
    st.caption(f"{len(gs)} garments · wardrobe: **{state.wardrobe().label}**. Change them on the Wardrobe and Garments pages.")
with right:
    st.subheader("Walk-through")
    st.markdown(
        "1. **Wardrobe** - pick an Indian wardrobe preset (single door to sliding) or edit the shelves.\n"
        "2. **Garments** - edit the family's clothes, or add one from a photo.\n"
        "3. **Plan** - run the optimizer, compare it with four rule-of-thumb baselines, read why each item went where it did.\n"
        "4. **Fold studio** - watch any garment fold in 3D, step by step.\n"
        "5. **Mechanism lab** - change what each person *says* their clothes are worth and see who gains by lying.\n"
        "6. **Experiments** - the seven experiments behind the numbers above."
    )
st.caption("Everything runs locally. No personal photos are used: demo images are rendered or from public datasets (see DATA.md).")
