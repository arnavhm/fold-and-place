"""Build synthetic households from ``configs/personas.yaml`` (fixed seeds, reproducible)."""

from __future__ import annotations

import zlib

import numpy as np

from fold_and_place import config
from fold_and_place.garments.record import Garment, make_garment


def persona_garments(persona: str, seed: int = 0, scale: float = 1.0) -> list[Garment]:
    """Garments owned by one persona. ``scale`` multiplies every item count (for size sweeps)."""
    spec = config.personas()[persona]
    # zlib.crc32 is stable across processes (built-in hash() of str is randomised per run).
    rng = np.random.default_rng(zlib.crc32(f"{persona}:{seed}".encode()))
    out: list[Garment] = []
    for category, count, size, v_lo, v_hi, wears in spec["items"]:
        n = max(0, int(round(count * scale)))
        for k in range(n):
            out.append(
                make_garment(
                    category,
                    f"{persona}-{category}-{k + 1}",
                    size,
                    owner=persona,
                    value=float(rng.integers(v_lo, v_hi + 1)),
                    wear_freq=float(wears),
                )
            )
    return out


def household_garments(
    personas: list[str] | None = None, seed: int = 0, scale: float = 1.0
) -> list[Garment]:
    names = personas or list(config.personas().keys())
    out: list[Garment] = []
    for p in names:
        out.extend(persona_garments(p, seed=seed, scale=scale))
    return out
