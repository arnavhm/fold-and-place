"""Load the YAML configs in ``configs/`` once and share them."""

from __future__ import annotations

from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"


@cache
def load(name: str) -> dict[str, Any]:
    """Return the parsed contents of ``configs/<name>.yaml``."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def categories() -> dict[str, dict[str, Any]]:
    return load("categories")["categories"]


def size_scale() -> dict[str, list[float]]:
    return load("categories")["size_scale"]


def fabrics() -> dict[str, dict[str, Any]]:
    return load("fabrics")["fabrics"]


_K_LOFT_OVERRIDE: list[float] = []


def k_loft() -> float:
    if _K_LOFT_OVERRIDE:
        return _K_LOFT_OVERRIDE[-1]
    return float(load("fabrics")["k_loft"])


@contextmanager
def override_k_loft(value: float):
    """Temporarily change the loft factor (sensitivity analysis in E4)."""
    _K_LOFT_OVERRIDE.append(float(value))
    try:
        yield
    finally:
        _K_LOFT_OVERRIDE.pop()


def presets() -> dict[str, dict[str, Any]]:
    return load("presets")["presets"]


def golden_zone() -> tuple[float, float]:
    lo, hi = load("presets")["golden_zone_cm"]
    return float(lo), float(hi)


def personas() -> dict[str, dict[str, Any]]:
    return load("personas")["personas"]
