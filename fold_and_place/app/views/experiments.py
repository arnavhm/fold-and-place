from __future__ import annotations

import pandas as pd
import streamlit as st

from fold_and_place import config
from fold_and_place.app import ui

ui.setup("Experiments", lead="The evidence behind the numbers. Re-run everything with <code>make experiments</code>; "
         "E2 also needs <code>make data embeddings</code>, E8 needs your own measured photos in <code>data/real/</code>.")

EXP = {
    "E1": "Fold geometry",
    "E2": "Recognising garments",
    "E3": "Measuring from photos",
    "E4": "Plan vs rules of thumb",
    "E5": "Sharing rules and lying",
    "E6": "Who pays for the wardrobe",
    "E7": "Everyone games the rule",
    "E8": "Real photos",
}
res = config.RESULTS_DIR
tabs = st.tabs([f"{k} · {v}" for k, v in EXP.items()])
for (key, _), tab in zip(EXP.items(), tabs):
    with tab:
        notes = res / f"{key}_notes.md"
        if notes.exists():
            st.markdown(notes.read_text())
        else:
            st.info("Not run yet.")
        pngs = sorted(res.glob(f"{key}_*.png"))
        for k in range(0, len(pngs), 2):
            cols = st.columns(2)
            for col, png in zip(cols, pngs[k : k + 2]):
                col.image(str(png), caption=png.stem.replace("_", " "), width="stretch")
        for csv in sorted(res.glob(f"{key}_*.csv")):
            with st.expander(f"Data: {csv.name}"):
                st.dataframe(pd.read_csv(csv), width="stretch", hide_index=True)
