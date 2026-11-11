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
