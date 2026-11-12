"""Fold & Place. Run: make app   (or: streamlit run fold_and_place/app/app.py)"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from fold_and_place.app import state

st.set_page_config(page_title="Fold & Place", page_icon="🧺", layout="wide")
state.init()

V = Path(__file__).parent / "views"
pages = {
    "": [st.Page(V / "start.py", title="Start", icon=":material/home:", default=True)],
    "Four steps": [
        st.Page(V / "wardrobe.py", title="1 · Your wardrobe", icon=":material/door_sliding:", url_path="wardrobe"),
        st.Page(V / "clothes.py", title="2 · Your clothes", icon=":material/apparel:", url_path="clothes"),
        st.Page(V / "plan.py", title="3 · Your plan", icon=":material/grid_view:", url_path="plan"),
        st.Page(V / "sharing.py", title="4 · Sharing fairly", icon=":material/balance:", url_path="sharing"),
    ],
    "Behind the scenes": [
        st.Page(V / "fold_studio.py", title="Fold any garment", icon=":material/view_in_ar:", url_path="fold"),
        st.Page(V / "experiments.py", title="Experiments", icon=":material/science:", url_path="experiments"),
        st.Page(V / "method.py", title="How it works", icon=":material/menu_book:", url_path="method"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    mode = st.session_state.get("mode", "demo")
    st.caption("Demo family loaded" if mode == "demo" else "Your own household")
    st.download_button("Save household", state.to_json(), file_name="household.json", mime="application/json",
                       width="stretch", help="Download everything you entered; load it again later.")
    up = st.file_uploader("Load a saved household", type=["json"], label_visibility="collapsed")
    if up is not None and st.session_state.get("_loaded") != up.file_id:
        state.from_json(up.getvalue().decode())
        st.session_state._loaded = up.file_id
        st.rerun()

nav.run()
