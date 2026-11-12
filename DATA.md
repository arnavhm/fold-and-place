# Data, models and licences

Nothing below is committed to this repository. Datasets and model weights are downloaded on your
own machine by the scripts and live in `data/` and `models/` (both git-ignored).

| What | Used for | Source | Licence / terms |
|---|---|---|---|
| IndoFashion (106K images, 15 Indian ethnic categories) | training + testing the classifier head (E2) | Rajput & Aneja, CVPRW 2021, <https://indofashion.github.io>, mirrored on Kaggle | research use; cite the paper |
| Fashion Product Images (Small), Myntra catalogue | training + testing (E2), Western and Indian product shots | Kaggle `paramaggarwal/fashion-product-images-small` | as stated on the Kaggle page |
| Marqo-FashionSigLIP | image/text embeddings, zero-shot classification | <https://huggingface.co/Marqo/marqo-fashionSigLIP> | Apache-2.0 |
| SegFormer-B2 clothes | garment masks on worn photos | <https://huggingface.co/mattmdjaga/segformer_b2_clothes> | MIT |
| rembg (U²-Net) | garment masks on flat-lay photos | <https://github.com/danielgatis/rembg> | MIT |

`assets/samples/` holds synthetic flat-lay images we RENDER ourselves (`scripts/make_samples.py`),
with their true sizes - no third-party images are included.

## Steps on your laptop

```bash
make setup        # installs everything, including the vision stack
make data         # Kaggle download (needs ~/.kaggle/kaggle.json); IndoFashion is several GB
make embeddings   # embeds a balanced sample, trains models/head.npz, writes results/E2_*
make smoke        # checks the real models on the bundled samples
```

If the IndoFashion Kaggle mirror has moved, run
`python scripts/fetch_data.py --indo-slug <owner>/<dataset>`; `make data` still fetches Myntra.

## Your own photos (E8, optional)

E8 checks the automatic measuring on real clothes. Photos stay on your machine: `data/` is
git-ignored, and the overlays E8 draws go to `data/real/out/`, never to `results/`.

1. Lay each garment flat on a plain sheet or floor, put **one A4 sheet** beside it (not touching,
   all four corners visible) and photograph it **from above**. Any rotation is fine.
   A few photos **without** the sheet test the approximate mode.
2. Measure each garment once with a tape, where the app measures:
   - T-shirts, shirts, men's kurtas, tops, jackets: width straight across the body about a hand-span
     above the hem; length from the top of the shoulder to the hem; sleeve from shoulder seam to cuff.
   - Kurtis, dresses, gowns: width across the chest just below the armpits; length shoulder to hem.
   - Jeans, trousers, palazzos, skirts: width across the waistband laid flat; length waistband to hem.
   - Dupatta: full width and length spread out.
3. Save the photos and a `measurements.csv` in `data/real/` (start from
   `assets/real_measurements_template.csv`; `category` uses the keys in `configs/categories.yaml`,
   e.g. `tshirt`, `kurti`, `jeans`).
4. Run `python scripts/run_experiments.py E8`. With the vision models installed it also checks the
   recognised garment type.

The bundled sample photos in the dashboard are rendered on your machine by
`fold_and_place/vision/synth.py` (textured bedsheet, print, creases, shadows, tilt, JPEG) and
know their true size; no third-party photos are shipped.
