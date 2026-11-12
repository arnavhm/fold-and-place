from __future__ import annotations

import hashlib

import streamlit as st

from fold_and_place import config
from fold_and_place.app import photo, state, ui
from fold_and_place.app.vision_status import vision_status
from fold_and_place.garments.record import make_garment

ui.setup("Your clothes", "clothes", "Add clothes from photos - lay the garment flat with a sheet of A4 paper "
         "beside it and the app measures it for you. No tape measure needed.")

cats = config.categories()
people = st.session_state.people
vs = vision_status()
tab_photo, tab_list = st.tabs([":material/photo_camera: Add from a photo", f":material/checkroom: Everyone's clothes ({len(state.garments())})"])


def cat_label(c: str) -> str:
    return cats[c]["label"]


# =================================================================================================
with tab_photo:
    with st.expander("How to take a good photo", expanded=not state.garments()):
        a, b = st.columns([3, 2])
        a.markdown(
            "1. Lay the garment **flat**, front up, on a **plain** sheet or floor that contrasts with it.\n"
            "2. Put **one A4 sheet** beside it - not touching, all four corners visible.\n"
            "3. Take the photo from **above**. A slight angle is fine; any rotation is fine.\n"
            "4. Good light, no strong shadow across the garment.\n\n"
            "No A4 sheet? It still works, but the size becomes an estimate based on the size label."
        )
        ui.raw(where=b, html=
            '<svg viewBox="0 0 220 150" width="100%" style="max-width:260px"><rect width="220" height="150" rx="10" fill="#4B3B5C"/>'
            '<path d="M40,30 L70,22 L100,30 L125,40 L118,58 L100,52 L100,125 L40,125 L40,52 L22,58 L15,40 Z" fill="#F2C14E"/>'
            '<rect x="150" y="70" width="42" height="58" fill="#FAFAF5" transform="rotate(8 171 99)"/>'
            '<text x="171" y="104" font-size="11" text-anchor="middle" fill="#7A6A58" transform="rotate(8 171 99)">A4</text></svg>'
        )

    src = st.segmented_control("Photo from", ["Upload", "Camera", "Sample photos"], default="Upload" if st.session_state.get("mode") == "own" else "Sample photos")
    files: list[tuple[str, bytes, tuple | None, str | None]] = []
    if src == "Upload":
        ups = st.file_uploader("Photos of your clothes", type=["jpg", "jpeg", "png", "webp", "heic"], accept_multiple_files=True,
                               help="You can pick several at once.")
        files = [(u.name, u.getvalue(), None, None) for u in ups or []]
    elif src == "Camera":
        shot = st.camera_input("Point at the garment and the A4 sheet")
        if shot:
            files = [("camera", shot.getvalue(), None, None)]
    elif src == "Sample photos":
        samples = photo.sample_photos()
        if not samples:
            st.info("Sample photos need OpenCV: run `make setup`.")
        else:
            names = [s["name"] for s in samples]
            pick = st.pills("Pick a sample (rendered on your computer, true size known)", names, default=names[0])
            s = next((x for x in samples if x["name"] == pick), None)
            if s:
                files = [(s["name"], s["bytes"], s["true"], s["category"])]

    if not vs["measure"]:
        st.warning("Measuring from photos needs OpenCV: run `make setup` once.")

    for _name, data, truth, hint in files:
        h = hashlib.md5(data).hexdigest()[:10]
        added = h in st.session_state.setdefault("_added_photos", set())
        with st.container(border=True):
            top = st.columns([1, 1, 1.25])
            size = "M"
            cat_key = f"cat_{h}"
            if not vs["recognise"] and cat_key not in st.session_state:
                st.session_state[cat_key] = hint or "tshirt"
            override = st.session_state.get(cat_key)
            try:
                with st.spinner("Looking at the photo..."):
                    guess, img = photo.analyse(data, st.session_state.get(f"size_{h}", "M"), override)
            except Exception as e:  # noqa: BLE001
                top[0].image(data, width="stretch")
                top[1].error(f"Couldn't read this photo ({type(e).__name__}: {e}).")
                continue
            top[0].image(photo.overlay(img, guess), caption="What the app saw: garment edge (orange), A4 sheet (green)", width="stretch")
            if guess.outline:
                ui.raw(where=top[1], html=photo.outline_svg(guess))
            else:
                top[1].info("No clear outline: the size comes from the size chart.")
            with top[2]:
                if vs["recognise"] and override is None:
                    c0, p0 = guess.top3[0]
                    st.markdown(f"### Looks like a **{cat_label(c0)}**")
                    sure = "very sure" if p0 > 0.7 else ("fairly sure" if p0 > 0.45 else "not sure")
                    st.caption(f"{sure} ({p0:.0%})" + (" · also possible: " + ", ".join(cat_label(c) for c, _ in guess.top3[1:]) if p0 < 0.7 else ""))
                else:
                    st.markdown(f"### {cat_label(guess.category)}")
                opts = [c for c, _ in guess.top3] + [c for c in cats if c not in {c for c, _ in guess.top3}]
                new = st.selectbox("Garment type", opts, index=0, format_func=cat_label, key=f"type_{h}")
                if new != guess.category:
                    st.session_state[cat_key] = new
                    st.rerun()
                if guess.size_source == "marker":
                    st.success(f"**{guess.flat_w:.0f} × {guess.flat_l:.0f} cm** - measured using your A4 sheet", icon=":material/straighten:")
                elif guess.size_source == "approx":
                    st.warning(f"About **{guess.flat_w:.0f} × {guess.flat_l:.0f} cm** - no A4 sheet found, so this is scaled to the size label", icon=":material/info:")
                    st.selectbox("Size on the label", list(config.size_scale()), index=1, key=f"size_{h}")
                else:
                    st.info(f"**{guess.flat_w:.0f} × {guess.flat_l:.0f} cm** - {guess.source_text}")
                if truth:
                    st.caption(f"True size of this sample: {truth[0]:.0f} × {truth[1]:.0f} cm "
                               f"(off by {guess.flat_w - truth[0]:+.1f} / {guess.flat_l - truth[1]:+.1f} cm)")
                if guess.sleeve_len:
                    st.caption(f"Sleeves about {guess.sleeve_len:.0f} cm")
                c1, c2 = st.columns(2)
                owner = c1.selectbox("Whose is it?", list(people), format_func=ui.person_label, key=f"own_{h}")
                wears = c2.select_slider("Worn", list(ui.WEAR_OPTIONS), value="Weekly", key=f"wear_{h}")
                val = st.select_slider("How much does it matter to them?", list(range(1, 11)), value=6,
                                       format_func=lambda v: f"{v} · {ui.VALUE_WORDS[v]}", key=f"val_{h}")
                if added:
                    st.success("Added ✓")
                elif st.button("Add to the wardrobe", type="primary", key=f"add_{h}", icon=":material/add:"):
                    gid = f"{owner}_{guess.category}_{len(state.garments()) + 1}"
                    g = guess.to_garment(gid, owner, float(val), ui.WEAR_OPTIONS[wears], st.session_state.get(f"size_{h}", "M"))
                    g.image = photo.thumb_uri(img)
                    state.set_garments(state.garments() + [g])
                    st.session_state._added_photos.add(h)
                    st.toast(f"Added {cat_label(guess.category)} for {ui.person_label(owner)}")
                    st.rerun()

# =================================================================================================
with tab_list:
    gs = state.garments()
    if not gs:
        st.info("No clothes yet. Add some from photos, or below without a photo.")
    who_opts = ["Everyone"] + list(people)
    who = st.segmented_control("Show", who_opts, default="Everyone", format_func=lambda k: k if k == "Everyone" else ui.person_label(k), key="who_list")
    shown = [g for g in gs if who in (None, "Everyone") or g.owner == who]
    summary = {}
    for g in shown:
        summary[g.label] = summary.get(g.label, 0) + 1
    if summary:
        st.caption(" · ".join(f"{k} × {v}" for k, v in sorted(summary.items(), key=lambda t: -t[1])))
    per_page = 24
    pages = max(1, (len(shown) + per_page - 1) // per_page)
    page = st.number_input("Page", 1, pages, 1, label_visibility="collapsed") if pages > 1 else 1
    chunk = shown[(page - 1) * per_page : page * per_page]
    for row in range(0, len(chunk), 6):
        cols = st.columns(6)
        for col, g in zip(cols, chunk[row : row + 6]):
            with col, st.container(border=True):
                src = {"marker": "📏 measured", "approx": "≈ estimated", "size_chart": "size chart", "fabric_standard": "standard size", "manual": "entered"}.get(g.size_source, "")
                ui.raw(f'<div style="display:flex;gap:.5rem;align-items:center">{ui.thumb(g, 46)}<div><b>{g.label}</b><br>{ui.chip(g.owner)}</div></div>'
                        f'<div class="fp-soft" style="margin-top:.35rem">worn {ui.wear_words(g.wear_freq)}<br>matters {ui.value_dots(g.value)}<br>{g.flat_w:.0f}×{g.flat_l:.0f} cm · {src}</div>')
                with st.popover("Edit", width="stretch"):
                    v = st.select_slider("Matters", list(range(1, 11)), value=int(round(g.value)), key=f"ev_{g.id}")
                    f = st.select_slider("Worn", list(ui.WEAR_OPTIONS), value=ui.nearest_wear(g.wear_freq), key=f"ef_{g.id}")
                    o = st.selectbox("Owner", list(people), index=list(people).index(g.owner) if g.owner in people else 0,
                                     format_func=ui.person_label, key=f"eo_{g.id}")
                    a, b = st.columns(2)
                    if a.button("Save", key=f"es_{g.id}", type="primary"):
                        g.value, g.wear_freq, g.owner = float(v), ui.WEAR_OPTIONS[f], o
                        state.set_garments(list(gs))
                        st.rerun()
                    if b.button("Remove", key=f"er_{g.id}"):
                        state.set_garments([x for x in gs if x.id != g.id])
                        st.rerun()

    with st.expander("Add a garment without a photo"):
        c1, c2, c3 = st.columns(3)
        cat = c1.selectbox("Type", list(cats), format_func=cat_label, key="np_cat")
        size = c2.selectbox("Size", list(config.size_scale()), index=1, key="np_size")
        owner = c3.selectbox("Whose", list(people), format_func=ui.person_label, key="np_owner")
        c4, c5 = st.columns(2)
        val = c4.select_slider("Matters", list(range(1, 11)), value=6, key="np_val")
        wears = c5.select_slider("Worn", list(ui.WEAR_OPTIONS), value="Monthly", key="np_wear")
        if st.button("Add", key="np_add"):
            g = make_garment(cat, f"{owner}_{cat}_{len(gs) + 1}", size, owner=owner, value=float(val), wear_freq=ui.WEAR_OPTIONS[wears])
            state.set_garments(gs + [g])
            st.rerun()

st.divider()
ui.next_button("Next: make the plan →", "views/plan.py")
