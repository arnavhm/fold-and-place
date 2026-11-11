from __future__ import annotations

import pandas as pd
import streamlit as st

from fold_and_place import config

st.set_page_config(page_title="Experiments · Fold & Place", page_icon="🧪", layout="wide")
st.title("Experiments")
st.caption("Regenerate everything with `make experiments` (E2 also needs `make data embeddings`). Files live in results/.")

EXP = {
    "E1": "Fold geometry and calibration",
    "E2": "Garment recognition (public datasets)",
    "E3": "Size from a photo (A4 sheet)",
    "E4": "Optimizer vs rules of thumb",
    "E5": "Mechanisms: efficiency, fairness, lying",
    "E6": "Sharing the cost of a bigger wardrobe (Shapley)",
    "E7": "Strategic play and price of anarchy",
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
        for png in sorted(res.glob(f"{key}_*.png")):
            st.image(str(png), caption=png.name, width="stretch")
        for csv in sorted(res.glob(f"{key}_*.csv")):
            with st.expander(csv.name):
                st.dataframe(pd.read_csv(csv), width="stretch", hide_index=True)
