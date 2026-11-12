"""E3 - How accurate are marker-based sizes?

Synthetic photos with KNOWN garment sizes: a garment outline (from our templates, in cm) and
an A4 sheet lie on a table; a pinhole camera looks down at a random tilt (perspective).
We rasterise the garment mask in the photo, add pixel noise to the detected sheet corners (as a
real detector would), measure through the homography and compare with the truth.
No real photos are used, so this measures the GEOMETRY of the method, not segmentation quality.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from fold_and_place.experiments.common import out_dir, save_csv, save_fig, write_notes
from fold_and_place.folding.templates import outline
from fold_and_place.garments import make_garment
from fold_and_place.vision.measure import apply_h, measure_with_marker

CASES = [("tshirt", "M"), ("shirt", "L"), ("jeans", "M"), ("trousers", "L"), ("skirt", "M"), ("kurti", "M")]
A4 = (21.0, 29.7)  # reference sheet lying next to the garment
IMG = (1600, 1200)


def camera(tilt_deg: float, yaw_deg: float, dist_cm: float, f_px: float, img=IMG):
    """3x3 homography from table plane (cm) to image pixels for a tilted pinhole camera."""
    t, y = np.radians(tilt_deg), np.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R = Rx @ Rz
    tvec = np.array([0.0, 0.0, dist_cm])
    K = np.array([[f_px, 0, img[0] / 2], [0, f_px, img[1] / 2], [0, 0, 1]])
    H = K @ np.c_[R[:, 0], R[:, 1], tvec]
    return H / H[2, 2]


def synth(category: str, size: str, rng: np.random.Generator, noise_px: float):
    """Garment (hem down) with an A4 sheet to its right, photographed from above at a tilt; the
    camera is framed so the whole scene fills about 80% of the photo."""
    g = make_garment(category, "e3", size)
    poly = np.asarray(outline(g).exterior.coords)[:-1]
    poly = np.c_[poly[:, 0], -(poly[:, 1] - g.flat_l / 2)]
    x_right = poly[:, 0].max() + 6
    sheet = np.array([[0, 0], [A4[0], 0], [A4[0], A4[1]], [0, A4[1]]]) + [x_right, -A4[1] / 2]
    scene = np.r_[poly, sheet]
    centre = (scene.min(axis=0) + scene.max(axis=0)) / 2
    poly, sheet = poly - centre, sheet - centre
    extent = np.ptp(np.r_[poly, sheet], axis=0)
    dist = 150.0
    f = 0.8 * min(IMG[0] / extent[0], IMG[1] / extent[1]) * dist
    H = camera(rng.uniform(0, 25), rng.uniform(-4, 4), dist_cm=dist, f_px=f)
    img_poly = apply_h(H, poly)
    img_sheet = apply_h(H, sheet) + rng.normal(0, noise_px, (4, 2))
    yy, xx = np.mgrid[0 : IMG[1], 0 : IMG[0]]
    mask = MplPath(img_poly).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(IMG[1], IMG[0])
    sheet_cm = np.array([[0, 0], [A4[0], 0], [A4[0], A4[1]], [0, A4[1]]])
    return g, mask, img_sheet, sheet_cm


REAL_CASES = ["tshirt", "shirt", "jeans", "trousers", "skirt", "kurti", "dress", "palazzo", "kurta_men"]


def realistic(d, per_case: int = 6) -> dict:
    """The whole real-photo chain on realistic synthetic photos: the A4 sheet must be FOUND
    (no corners given), the garment segmented by the plain-background segmenter (rembg is better,
    but needs a download), turned upright from any rotation, measured. Plus the no-sheet
    'approximate' mode with the correct size label."""
    try:
        from fold_and_place.vision.reference import detect_a4, main_part
        from fold_and_place.vision.synth import render
    except ImportError:  # pragma: no cover
        return {"skipped": "needs OpenCV"}
    from fold_and_place.vision.measure import drop_marker, homography
    from fold_and_place.vision.models import PlainSegmenter
    from fold_and_place.vision.shape import silhouette

    seg = PlainSegmenter()
    rows = []
    for ci, cat in enumerate(REAL_CASES):
        for k in range(per_case):
            seed = 1000 + 37 * ci + k
            for sheet in (True, False):
                ph = render(cat, "M", seed=seed, sheet=sheet)
                g = ph.garment
                ref = detect_a4(ph.image) if sheet else None
                mask = seg(ph.image)
                if ref is not None:
                    mask = drop_marker(mask, ref[0], grow_px=max(6.0, 0.05 * np.linalg.norm(ref[0][1] - ref[0][0])))
                mask = main_part(mask)
                row = {"category": cat, "seed": seed, "sheet_in_photo": sheet, "sheet_found": ref is not None,
                       "true_w": g.flat_w, "true_l": g.flat_l}
                try:
                    if sheet and ref is None:
                        raise ValueError("sheet not found")
                    s = silhouette(mask, g.template, homography(ref[0], ref[1]) if ref is not None else None, length_cm=g.flat_l)
                    row.update(meas_w=round(s.measured.flat_w, 2), meas_l=round(s.measured.flat_l, 2))
                except ValueError:
                    row.update(meas_w=np.nan, meas_l=np.nan)
                rows.append(row)
    df = pd.DataFrame(rows)
    df["err_w_cm"] = (df.meas_w - df.true_w).round(2)
    df["err_l_cm"] = (df.meas_l - df.true_l).round(2)
    save_csv(df, d, "E3_realistic.csv")
    withs, nos = df[df.sheet_in_photo], df[~df.sheet_in_photo]
    found = withs[withs.sheet_found]
    out = {
        "photos_with_sheet": len(withs),
        "sheet_found_pct": round(100 * withs.sheet_found.mean(), 1),
        "measured_mae_w_cm": round(float(found.err_w_cm.abs().mean()), 2),
        "measured_mae_l_cm": round(float(found.err_l_cm.abs().mean()), 2),
        "measured_median_w_cm": round(float(found.err_w_cm.abs().median()), 2),
        "measured_median_l_cm": round(float(found.err_l_cm.abs().median()), 2),
        "within_3cm_pct": round(100 * float(((found.err_w_cm.abs() <= 3) & (found.err_l_cm.abs() <= 3)).mean()), 1),
        "approx_mae_w_cm": round(float(nos.err_w_cm.abs().mean()), 2),
        "approx_rel_w_pct": round(float((100 * nos.err_w_cm.abs() / nos.true_w).mean()), 1),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, (name, sub) in zip(axes, (("A4 sheet found automatically", found), ("no sheet: approximate", nos))):
        ax.scatter(sub.true_w, sub.err_w_cm, s=14, alpha=0.7, label="width")
        if name.startswith("A4"):
            ax.scatter(sub.true_l, sub.err_l_cm, s=14, alpha=0.7, label="length")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("True size (cm)")
        ax.set_ylabel("Measured - true (cm)")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
    save_fig(fig, d, "E3_realistic.png")
    return out


def run(path=None, trials: int = 12, seed: int = 0) -> dict:
    d = out_dir(path)
    real = realistic(d)
    rng = np.random.default_rng(seed)
    rows = []
    for noise in (0.5, 1.5):
        for cat, size in CASES:
            for _ in range(trials):
                g, mask, mpx, mcm = synth(cat, size, rng, noise)
                if mask.sum() < 500 or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                    continue  # garment not fully in the photo
                m = measure_with_marker(mask, mpx, mcm, g.template)
                rows.append(
                    {
                        "category": cat,
                        "corner_noise_px": noise,
                        "true_w": g.flat_w,
                        "true_l": g.flat_l,
                        "meas_w": round(m.flat_w, 2),
                        "meas_l": round(m.flat_l, 2),
                        "err_w_cm": round(m.flat_w - g.flat_w, 2),
                        "err_l_cm": round(m.flat_l - g.flat_l, 2),
                    }
                )
    df = pd.DataFrame(rows)
    save_csv(df, d, "E3_measure.csv")
    summ = (
        df.assign(abs_w=df.err_w_cm.abs(), abs_l=df.err_l_cm.abs(), rel_l=100 * df.err_l_cm.abs() / df.true_l)
        .groupby(["corner_noise_px", "category"])[["abs_w", "abs_l", "rel_l"]]
        .mean()
        .round(2)
        .reset_index()
    )
    save_csv(summ, d, "E3_summary.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    for noise, mk in ((0.5, "o"), (1.5, "s")):
        s = df[df.corner_noise_px == noise]
        ax.scatter(s.true_l, s.err_l_cm, s=12, marker=mk, alpha=0.6, label=f"length, corner noise {noise} px")
        ax.scatter(s.true_w, s.err_w_cm, s=12, marker=mk, alpha=0.6, label=f"width, corner noise {noise} px")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("True size (cm)")
    ax.set_ylabel("Measured - true (cm)")
    ax.legend(fontsize=7)
    ax.set_title("E3: A4 sheet + homography on synthetic tilted photos", fontsize=9)
    save_fig(fig, d, "E3_measure.png")
    out = {
        "mean_abs_err_w_cm": round(float(df.err_w_cm.abs().mean()), 2),
        "mean_abs_err_l_cm": round(float(df.err_l_cm.abs().mean()), 2),
        "p95_abs_err_cm": round(float(np.percentile(np.r_[df.err_w_cm.abs(), df.err_l_cm.abs()], 95)), 2),
        "photos": len(df),
        "realistic": real,
    }
    write_notes(
        d,
        "E3_notes.md",
        f"""
# E3 - Marker-based size accuracy (synthetic)

{out['photos']} synthetic photos ({len(CASES)} garments, camera tilt 0-25 deg, A4-sheet corner noise 0.5
and 1.5 px). Mean absolute error: width {out['mean_abs_err_w_cm']} cm, length {out['mean_abs_err_l_cm']} cm;
95th percentile {out['p95_abs_err_cm']} cm. See E3_summary.csv per garment.

## The whole chain on realistic photos

{real.get('photos_with_sheet', 0)} realistic synthetic photos (textured bedsheet, prints, creases, shadows,
uneven light, any rotation, tilt up to 25 deg, JPEG), the A4 sheet **found automatically**, the garment
segmented by the plain-background segmenter and turned upright: sheet found in
{real.get('sheet_found_pct', 'n/a')}% of photos; mean absolute error width {real.get('measured_mae_w_cm', 'n/a')} cm,
length {real.get('measured_mae_l_cm', 'n/a')} cm (median {real.get('measured_median_w_cm', 'n/a')} /
{real.get('measured_median_l_cm', 'n/a')} cm); {real.get('within_3cm_pct', 'n/a')}% within 3 cm on both. Without a
sheet, scaled to the correct size label, width error {real.get('approx_mae_w_cm', 'n/a')} cm
({real.get('approx_rel_w_pct', 'n/a')}%; this mode cannot undo camera tilt, so shoot from straight above).
Segmenting a plain-background photo is not the bottleneck here: as in the perfect-mask test, the error is
driven by how precisely the sheet's corners are located relative to the sheet's size in the photo (a small,
distant sheet magnifies corner error). The synthetic garments have wrinkled edges, which adds a little.
E8 repeats this on real photos.

For comparison, published garment-measurement systems report about 1 cm average error on real
photos (Paulauskaite-Taraseviciene et al., Applied Sciences 2022). Our synthetic test has a perfect
mask, so real-photo error will be higher: segmentation is the dominant error source, not geometry.
Stock photos carry no marker; their sizes come from size charts / fabric standards and are labelled so.
""",
    )
    return out


if __name__ == "__main__":
    print(run())
