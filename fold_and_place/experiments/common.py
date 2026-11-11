"""Shared helpers for experiment runners."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fold_and_place.config import RESULTS_DIR  # noqa: E402


def out_dir(path: str | Path | None = None) -> Path:
    p = Path(path) if path else RESULTS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, directory: Path, name: str) -> Path:
    p = directory / name
    df.to_csv(p, index=False)
    return p


def save_fig(fig: plt.Figure, directory: Path, name: str) -> Path:
    p = directory / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def write_notes(directory: Path, name: str, text: str) -> Path:
    p = directory / name
    p.write_text(text.strip() + "\n", encoding="utf-8")
    return p
