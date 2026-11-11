"""Check that the real vision models load and run (downloads weights on first run).

    python scripts/smoke_vision.py

Analyses the bundled synthetic samples (measured with their A4 sheet) and, if data/ exists, a few
dataset images; prints category, photo type, size and size source for each.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fold_and_place.vision import Analyzer  # noqa: E402


def main() -> int:
    from PIL import Image

    t = time.time()
    an = Analyzer()
    meta = json.loads((ROOT / "assets" / "samples" / "samples.json").read_text())
    for m in meta:
        img = Image.open(ROOT / "assets" / "samples" / m["file"])
        marker = (np.array(m["a4_corners_px"]), np.array(m["a4_corners_cm"]))
        g = an.analyse(img, size=m["size"], marker=marker)
        print(
            f"{m['file']:18s} -> {g.category:12s} ({g.top3[0][1]:.2f}), {g.photo_type:6s}, "
            f"{g.flat_w:.1f} x {g.flat_l:.1f} cm [{g.size_source}]  true {m['true_flat_w_cm']:.1f} x {m['true_flat_l_cm']:.1f}"
        )
    extra = list((ROOT / "data").rglob("*.jpg"))[:3] if (ROOT / "data").exists() else []
    for p in extra:
        g = an.analyse(Image.open(p))
        print(f"{p.name:18s} -> {g.category:12s} ({g.top3[0][1]:.2f}), {g.photo_type:6s}, {g.flat_w:.0f} x {g.flat_l:.0f} cm [{g.size_source}]")
    print(f"ok in {time.time() - t:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
