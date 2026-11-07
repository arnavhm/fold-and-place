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


def run(path=None, trials: int = 12, seed: int = 0) -> dict:
    d = out_dir(path)
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
    }
    write_notes(
        d,
        "E3_notes.md",
        f"""
# E3 - Marker-based size accuracy (synthetic)

{out['photos']} synthetic photos ({len(CASES)} garments, camera tilt 0-25 deg, A4-sheet corner noise 0.5
and 1.5 px). Mean absolute error: width {out['mean_abs_err_w_cm']} cm, length {out['mean_abs_err_l_cm']} cm;
95th percentile {out['p95_abs_err_cm']} cm. See E3_summary.csv per garment.

For comparison, published garment-measurement systems report about 1 cm average error on real
photos (Paulauskaite-Taraseviciene et al., Applied Sciences 2022). Our synthetic test has a perfect
mask, so real-photo error will be higher: segmentation is the dominant error source, not geometry.
Stock photos carry no marker; their sizes come from size charts / fabric standards and are labelled so.
""",
    )
    return out


if __name__ == "__main__":
    print(run())
