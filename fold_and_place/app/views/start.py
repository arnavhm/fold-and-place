from __future__ import annotations

import streamlit as st

from fold_and_place.app import state, ui
from fold_and_place.app.vision_status import vision_status

ui.setup("Fold & Place")
ui.raw(
    '<div class="fp-hero-sub">Photograph your clothes and it works out <b>how to fold each one</b>, '
    "<b>where it goes</b> in your almirah, and <b>how to share the space fairly</b> when everyone "
    "thinks their clothes matter most.</div>"
)
st.write("")

cards = [
    ("1", "Your wardrobe", "Pick your almirah and who uses it. Heights decide what counts as easy to reach.", "views/wardrobe.py"),
    ("2", "Your clothes", "Snap a photo with an A4 sheet beside the garment. It recognises the garment and measures it for you.", "views/clothes.py"),
    ("3", "Your plan", "Every garment gets a fold and a place. Click a shelf to see what's in it and watch the folds.", "views/plan.py"),
    ("4", "Sharing fairly", "When not everything fits: whose clothes get the space, and can anyone cheat?", "views/sharing.py"),
]
cols = st.columns(4)
for col, (n, title, text, page) in zip(cols, cards):
    with col, st.container(border=True, height=210):
        st.markdown(f"**{n} · {title}**")
        st.caption(text)
        st.page_link(page, label="Open", icon=":material/arrow_forward:")

st.write("")
left, right = st.columns([3, 2])
with left:
    st.subheader("How do you want to start?")
    a, b = st.columns(2)
    with a, st.container(border=True):
        st.markdown("**Demo family**")
        st.caption("A student, a mother and a father with 97 clothes, Indian and Western. Everything is ready to explore.")
        if st.button("Use the demo family", type="primary", width="stretch"):
            state.load_demo()
            st.switch_page("views/wardrobe.py")
    with b, st.container(border=True):
        st.markdown("**My own clothes**")
        st.caption("Start empty and add your clothes from photos.")
        name = st.text_input("Your name", "Me", label_visibility="collapsed", placeholder="Your name")
        if st.button("Start with my clothes", width="stretch"):
            state.start_fresh(name or "Me")
            st.switch_page("views/wardrobe.py")
with right:
    st.subheader("Ready on this computer")
    vs = vision_status()
    st.markdown(
        f"{'✅' if vs['measure'] else '⚠️'} **Measuring from photos** (A4 sheet detection)  \n"
        f"{'✅' if vs['recognise'] else '⚠️'} **Recognising garments** {'(fashion model)' if vs['recognise'] else '- not installed: you pick the type'}  \n"
        "✅ **Folding, planning and fair sharing**"
    )
    if not vs["recognise"]:
        st.caption("To recognise garment types automatically, run `make setup` once (about 2 GB).")
    mode = st.session_state.get("mode")
    st.caption(f"Now loaded: **{'demo family' if mode == 'demo' else 'your household'}** · "
               f"{len(state.garments())} clothes · {state.wardrobe().label}")
