from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from fold_and_place import config
from fold_and_place.app import state
from fold_and_place.garments.record import make_garment

st.set_page_config(page_title="Garments · Fold & Place", page_icon="👕", layout="wide")
state.init()
st.title("Garments")

cats = config.categories()
SAMPLES = config.ROOT / "assets" / "samples"
tab_list, tab_photo = st.tabs(["Household list", "Add from a photo"])

# ---------------------------------------------------------------------------------------------
with tab_list:
    gs = state.garments()
    owners = list(config.personas()) + ["guest"]
    df = pd.DataFrame(
        [
            {
                "id": g.id,
                "owner": g.owner,
                "category": g.category,
                "size": g.size,
                "value (1-10)": g.value,
                "wears / month": g.wear_freq,
                "flat w x l (cm)": f"{g.flat_w:.0f} x {g.flat_l:.0f}",
                "fabric": g.fabric,
                "size from": g.size_source,
            }
            for g in gs
        ]
    )
    f1, f2 = st.columns([1, 3])
    who = f1.multiselect("Owner", sorted(df.owner.unique()), default=sorted(df.owner.unique()))
    f2.caption(
        "Values and wear rates of the demo family are labelled assumptions (configs/personas.yaml). "
        "Edit cells, add rows at the bottom, then press **Apply**. Sizes come from the category's size chart "
        "unless a photo with an A4 sheet measured them."
    )
    view = df[df.owner.isin(who)]
    ed = st.data_editor(
        view,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        disabled=["id", "flat w x l (cm)", "fabric", "size from"],
        column_config={
            "owner": st.column_config.SelectboxColumn(options=owners, required=True),
            "category": st.column_config.SelectboxColumn(options=list(cats), required=True),
            "size": st.column_config.SelectboxColumn(options=list(config.size_scale()), required=True),
            "value (1-10)": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=0.5),
            "wears / month": st.column_config.NumberColumn(min_value=0.0, max_value=30.0, step=0.1),
        },
        height=420,
        key="garment_editor",
    )
    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("Apply", type="primary"):
        old = {g.id: g for g in gs}
        kept = [g for g in gs if g.owner not in who]  # rows hidden by the filter stay as they were
        n = len(gs)
        for r in ed.to_dict("records"):
            if not r.get("category") or not r.get("owner"):
                continue
            gid = r.get("id") if isinstance(r.get("id"), str) and r.get("id") else f"{r['owner']}_{r['category']}_{n}"
            n += 1
            prev = old.get(gid)
            same_shape = prev is not None and prev.category == r["category"] and prev.size == r["size"]
            g = make_garment(
                r["category"], gid, r["size"], owner=r["owner"],
                value=float(r["value (1-10)"] if r["value (1-10)"] == r["value (1-10)"] else 5.0),
                wear_freq=float(r["wears / month"] if r["wears / month"] == r["wears / month"] else 1.0),
                fabric=prev.fabric if same_shape else None,
                embellish=prev.embellish if same_shape else None,
                flat_w=prev.flat_w if same_shape else None,
                flat_l=prev.flat_l if same_shape else None,
                size_source=prev.size_source if same_shape else None,
                confidence=prev.confidence if same_shape else 1.0,
            )
            kept.append(g)
        st.session_state.garments = kept
        st.session_state.plan = None
        st.success(f"{len(kept)} garments. Re-run the plan.")
        st.rerun()
    if c2.button("Reset demo family"):
        from fold_and_place.garments import household_garments

        st.session_state.garments = household_garments()
        st.session_state.plan = None
        st.rerun()
    st.markdown("##### Mix by owner and category")
    st.dataframe(pd.crosstab(df.category, df.owner), width="stretch", height=300)

# ---------------------------------------------------------------------------------------------
with tab_photo:
    st.markdown(
        "Photograph a garment **flat** with a known reference in the frame (an A4 sheet, or a printed ArUco marker that "
        "is found automatically) for a measured size - or use any "
        "catalogue/worn photo and the size comes from the category's size chart. The demo samples below are "
        "rendered images with known true sizes."
    )
    meta = json.loads((SAMPLES / "samples.json").read_text()) if (SAMPLES / "samples.json").exists() else []
    src = st.radio("Image", ["Sample", "Upload"], horizontal=True)
    img, marker, truth = None, None, None
    if src == "Sample" and meta:
        s = st.selectbox("Sample", meta, format_func=lambda m: f"{m['file']} (true {m['true_flat_w_cm']:.0f} x {m['true_flat_l_cm']:.0f} cm)")
        from PIL import Image

        img = Image.open(SAMPLES / s["file"]).convert("RGB")
        marker = (np.asarray(s["a4_corners_px"], float), np.asarray(s["a4_corners_cm"], float))
        truth = s
    elif src == "Upload":
        up = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            from PIL import Image

            img = Image.open(up).convert("RGB")
            if st.checkbox("Look for a printed 10 cm ArUco marker (DICT_4X4_50)", value=True):
                try:
                    from fold_and_place.vision.measure import detect_marker

                    marker = detect_marker(np.ascontiguousarray(np.asarray(img)[:, :, ::-1]))
                    st.caption("Marker found." if marker is not None else "No marker found: size will come from the size chart.")
                except ImportError:
                    st.caption("OpenCV not installed (pip install -e '.[vision]'): no marker detection.")
    size = st.selectbox("Labelled size (used when there is no A4 sheet)", list(config.size_scale()), index=1)
    owner = st.selectbox("Owner", list(config.personas()))

    if img is not None:
        left, r = st.columns([1, 1])
        left.image(img, width="stretch")
        with r:
            if st.button("Analyse", type="primary"):
                try:
                    from fold_and_place.vision import Analyzer

                    @st.cache_resource(show_spinner="Loading vision models (first time downloads ~1 GB)...")
                    def analyzer():
                        return Analyzer()

                    with st.spinner("Classifying, segmenting, measuring..."):
                        guess = analyzer().analyse(img, size=size, marker=marker)
                    st.session_state.guess = guess
                except Exception as e:  # missing extras / no network for weights
                    st.session_state.guess = None
                    st.error(f"Vision models unavailable here ({type(e).__name__}: {e}). "
                             "Install with `make setup` on a machine with internet; the rest of the dashboard works without them.")
            guess = st.session_state.get("guess")
            if guess is not None:
                st.markdown(f"**{cats[guess.category]['label']}** ({guess.photo_type} photo)")
                st.dataframe(pd.DataFrame([(cats[c]["label"], f"{p:.0%}") for c, p in guess.top3], columns=["category", "probability"]), hide_index=True)
                st.markdown(
                    f"Fabric **{guess.fabric}**, embellishment **{guess.embellish}**  \n"
                    f"Flat size **{guess.flat_w:.1f} x {guess.flat_l:.1f} cm** (from {guess.size_source}), confidence {guess.confidence:.0%}"
                )
                if truth:
                    st.caption(f"True size {truth['true_flat_w_cm']:.0f} x {truth['true_flat_l_cm']:.0f} cm "
                               f"(error {guess.flat_w - truth['true_flat_w_cm']:+.1f} / {guess.flat_l - truth['true_flat_l_cm']:+.1f} cm).")
                if guess.mask is not None:
                    st.image((guess.mask > 0).astype(np.uint8) * 255, caption="garment mask", width=260)
                value = st.slider("How much does the owner value it?", 1.0, 10.0, 6.0, 0.5)
                wears = st.number_input("Wears per month", 0.0, 30.0, 2.0, 0.5)
                if st.button("Add to household"):
                    gid = f"{owner}_{guess.category}_photo{len(state.garments())}"
                    st.session_state.garments = state.garments() + [guess.to_garment(gid, owner, value, wears, size)]
                    st.session_state.plan = None
                    st.success(f"Added {gid}.")
