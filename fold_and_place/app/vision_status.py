"""What the photo features can do on this machine, without loading any model."""

from __future__ import annotations

import functools
import importlib.util


@functools.cache
def vision_status() -> dict[str, bool]:
    has = lambda m: importlib.util.find_spec(m) is not None  # noqa: E731
    return {
        "measure": has("cv2") and has("scipy"),
        "recognise": has("torch") and has("open_clip"),
        "segment": has("rembg"),
    }
