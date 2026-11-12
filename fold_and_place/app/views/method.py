from __future__ import annotations

import re

import streamlit as st

from fold_and_place import config
from fold_and_place.app import ui

ui.setup("How it works", lead="The method, from photo to fair shares. This is <code>docs/design.md</code>.")
doc = config.ROOT / "docs" / "design.md"
if doc.exists():
    text = re.sub(r"^# Design\s*", "", doc.read_text())
    st.markdown(text)
else:
    st.info("docs/design.md not found.")
