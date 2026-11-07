"""Download the public training data with the Kaggle API (run on your laptop).

Needs a Kaggle account and API token (~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY).
Datasets (research use; see DATA.md - nothing is redistributed or committed):

  IndoFashion  - 106K images, 15 Indian ethnic categories (Rajput & Aneja, CVPRW 2021).
                 Large download (several GB).
  Myntra small - "Fashion Product Images (Small)", 44K product photos with styles.csv
                 (paramaggarwal/fashion-product-images-small, ~0.6 GB).

Usage:
    python scripts/fetch_data.py                   # both
    python scripts/fetch_data.py --only myntra     # just the small one
    python scripts/fetch_data.py --indo-slug owner/indo-fashion-dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MYNTRA_SLUG = "paramaggarwal/fashion-product-images-small"
# The IndoFashion upload's owner on Kaggle; tried in order. Pass --indo-slug to override.
INDO_SLUGS = ["validmodel/indo-fashion-dataset"]


def _api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("The 'kaggle' package is missing: run `make setup` (or pip install kaggle).")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Kaggle authentication failed ({exc}). Put your API token in ~/.kaggle/kaggle.json.")
    return api


def _download(api, slug: str, dest: Path) -> bool:
    if dest.exists() and any(dest.iterdir()):
        print(f"[skip] {dest} already has files")
        return True
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] {slug} -> {dest} (this can take a while)")
    try:
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] {slug}: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["indofashion", "myntra"])
    ap.add_argument("--indo-slug", help="Kaggle slug for IndoFashion (owner/dataset)")
    args = ap.parse_args()
    api = _api()
    ok = True
    if args.only in (None, "myntra"):
        ok &= _download(api, MYNTRA_SLUG, DATA / "myntra")
    if args.only in (None, "indofashion"):
        slugs = [args.indo_slug] if args.indo_slug else INDO_SLUGS
        got = any(_download(api, s, DATA / "indofashion") for s in slugs)
        if not got:
            print("IndoFashion not downloaded. Search Kaggle for 'indo fashion dataset' and pass --indo-slug.")
        ok &= got
    print("done" if ok else "finished with problems (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
