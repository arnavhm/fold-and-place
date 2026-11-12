"""E8 - Real photos: does the automatic measuring hold up on real clothes?

Put phone photos and one measurements file in ``data/real/`` (never committed):

    data/real/measurements.csv   columns: photo, category, size, width_cm, length_cm, sleeve_cm, notes
    data/real/<photo files>

``category`` is one of the 31 category keys (tshirt, kurti, jeans, ...); ``width_cm`` and
``length_cm`` are tape measurements taken once for this test (see DATA.md for where to measure).
Photos without an A4 sheet are allowed: they test the approximate mode.

For each photo we run the same chain as the app: find the sheet, segment, turn upright, measure;
with the recognition models installed we also check the predicted category. Overlays (photo with
the found sheet and garment edge) go to ``data/real/out/`` - next to the photos, not into results/,
because they show your clothes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fold_and_place.config import ROOT
from fold_and_place.experiments.common import out_dir, save_csv, write_notes

REAL = ROOT / "data" / "real"
COLUMNS = ["photo", "category", "size", "width_cm", "length_cm", "sleeve_cm", "notes"]


def run(path=None) -> dict:
    d = out_dir(path)
    sheet = REAL / "measurements.csv"
    if not sheet.exists():
        write_notes(d, "E8_notes.md", f"""
# E8 - Real photos

Not run yet: put photos and `measurements.csv` ({', '.join(COLUMNS)}) in `data/real/`, then
`python scripts/run_experiments.py E8`. A template is in `assets/real_measurements_template.csv`.
""")
        return {"skipped": "no data/real/measurements.csv"}
    from PIL import Image

    from fold_and_place.app.vision_status import vision_status
    from fold_and_place.vision.pipeline import measure_photo

    truth = pd.read_csv(sheet)
    truth.columns = [c.strip().lower() for c in truth.columns]
    recognise = vision_status()["recognise"]
    analyzer = None
    if recognise:
        from fold_and_place.vision import Analyzer

        analyzer = Analyzer()
    (REAL / "out").mkdir(exist_ok=True)
    rows = []
    for r in truth.itertuples(index=False):
        f = REAL / str(r.photo)
        if not f.exists():
            rows.append({"photo": r.photo, "error": "file not found"})
            continue
        img = Image.open(f).convert("RGB")
        img.thumbnail((2000, 2000))
        size = str(getattr(r, "size", "M") or "M")
        size = size if size in ("S", "M", "L", "XL") else "M"
        row = {"photo": r.photo, "category": r.category, "true_w": r.width_cm, "true_l": r.length_cm}
        try:
            g = measure_photo(img, r.category, size)
            if analyzer is not None:
                pred = analyzer.analyse(img, size=size)
                row.update(predicted=pred.category, top3=[c for c, _ in pred.top3], correct=pred.category == r.category,
                           in_top3=r.category in [c for c, _ in pred.top3])
            row.update(sheet_found=g.reference is not None, source=g.size_source, meas_w=g.flat_w, meas_l=g.flat_l,
                       meas_sleeve=g.sleeve_len)
            _overlay(img, g).save(REAL / "out" / f"{f.stem}_seen.jpg", quality=85)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    df = pd.DataFrame(rows)
    if "meas_w" in df:
        df["err_w_cm"] = (df.meas_w - df.true_w).round(1)
        df["err_l_cm"] = (df.meas_l - df.true_l).round(1)
    save_csv(df.drop(columns=[c for c in ("top3",) if c in df]), d, "E8_real.csv")
    meas = df[df.get("source", pd.Series(dtype=str)) == "marker"] if "source" in df else df.iloc[0:0]
    approx = df[df.get("source", pd.Series(dtype=str)) == "approx"] if "source" in df else df.iloc[0:0]
    out = {
        "photos": len(df),
        "with_sheet_found": int(len(meas)),
        "measured_mae_w_cm": round(float(meas.err_w_cm.abs().mean()), 2) if len(meas) else None,
        "measured_mae_l_cm": round(float(meas.err_l_cm.abs().mean()), 2) if len(meas) else None,
        "measured_within_3cm_pct": round(100 * float(((meas.err_w_cm.abs() <= 3) & (meas.err_l_cm.abs() <= 3)).mean()), 1) if len(meas) else None,
        "approx_mae_w_cm": round(float(approx.err_w_cm.abs().mean()), 2) if len(approx) else None,
        "approx_mae_l_cm": round(float(approx.err_l_cm.abs().mean()), 2) if len(approx) else None,
        "category_accuracy_pct": round(100 * float(df.correct.mean()), 1) if "correct" in df else None,
        "category_top3_pct": round(100 * float(df.in_top3.mean()), 1) if "in_top3" in df else None,
        "errors": int(df["error"].notna().sum()) if "error" in df else 0,
    }
    write_notes(d, "E8_notes.md", f"""
# E8 - Real photos

{out['photos']} real phone photos of garments laid flat; true sizes measured once with a tape.

- A4 sheet found automatically in **{out['with_sheet_found']}** photos. Measured size error (mean absolute):
  width **{out['measured_mae_w_cm']} cm**, length **{out['measured_mae_l_cm']} cm**;
  {out['measured_within_3cm_pct']}% within 3 cm on both.
- Without a sheet (approximate, scaled to the size label): width {out['approx_mae_w_cm']} cm,
  length {out['approx_mae_l_cm']} cm.
- Garment type: {'correct for ' + str(out['category_accuracy_pct']) + '% (in the top 3 for ' + str(out['category_top3_pct']) + '%)' if out['category_accuracy_pct'] is not None else 'not checked (recognition models not installed)'}.
- Photos that failed: {out['errors']}.

Per-photo results: E8_real.csv. Overlays showing what the app saw are in data/real/out/ (not committed).
""")
    return out


def _overlay(img, g):
    from PIL import ImageDraw

    im = img.copy()
    dr = ImageDraw.Draw(im, "RGBA")
    if g.mask is not None:
        try:
            import cv2

            cs, _ = cv2.findContours(g.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if len(c) > 10:
                    pts = [tuple(map(int, p[0])) for p in c]
                    dr.line(pts + [pts[0]], fill=(232, 120, 40, 255), width=max(3, im.width // 300))
        except ImportError:
            pass
    if g.reference is not None:
        dr.polygon([tuple(map(float, p)) for p in g.reference[0]], fill=(60, 170, 90, 70), outline=(40, 150, 70, 255))
    dr.text((10, 10), f"{g.category}: {g.flat_w:.0f} x {g.flat_l:.0f} cm ({g.size_source})", fill=(255, 255, 255, 255))
    return im


if __name__ == "__main__":
    print(run())
