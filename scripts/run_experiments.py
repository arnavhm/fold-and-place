"""Run experiments and write results/.

Usage:
    python scripts/run_experiments.py            # all available experiments
    python scripts/run_experiments.py E1 E4      # selected ones
"""

from __future__ import annotations

import importlib
import json
import sys
import time

EXPERIMENTS = {
    "E1": "fold_and_place.experiments.e1_folds",
    "E2": "fold_and_place.experiments.e2_vision",
    "E3": "fold_and_place.experiments.e3_measure",
    "E4": "fold_and_place.experiments.e4_optimizer",
    "E5": "fold_and_place.experiments.e5_mechanisms",
    "E6": "fold_and_place.experiments.e6_shapley",
    "E7": "fold_and_place.experiments.e7_poa",
}


def main(argv: list[str]) -> int:
    wanted = [a.upper() for a in argv] or list(EXPERIMENTS)
    status = 0
    for key in wanted:
        if key not in EXPERIMENTS:
            print(f"{key}: not available yet (have {', '.join(EXPERIMENTS)})")
            status = 1
            continue
        t = time.time()
        mod = importlib.import_module(EXPERIMENTS[key])
        summary = mod.run()
        print(f"{key} done in {time.time() - t:.1f}s")
        print(json.dumps(summary, indent=2, default=str))
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
