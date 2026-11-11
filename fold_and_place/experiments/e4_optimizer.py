"""E4 - Does the optimizer beat common practice?

1. Every preset x household size: optimizer vs B1 traditional, B2 KonMari, B3 hang-first, B4 greedy.
2. Trade-off sweep on a scarce wardrobe: weight on retrieval effort vs value stored.
3. Loft-factor sensitivity (+-30%): does the conclusion survive a wrong thickness model?
Every plan is re-verified by the independent checker before it is counted.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fold_and_place import config
from fold_and_place.experiments.common import out_dir, save_csv, save_fig, write_notes
from fold_and_place.garments import household_garments
from fold_and_place.storage import SolveConfig, assert_valid, load_preset, solve
from fold_and_place.storage.baselines import BASELINES

PRESETS = ["one_door", "two_door", "three_door", "sliding"]
SCALES = [1.0, 2.0]
TIME_LIMIT = 30.0


def _compare(time_limit: float = TIME_LIMIT) -> pd.DataFrame:
    rows = []
    for preset in PRESETS:
        w = load_preset(preset)
        for scale in SCALES:
            gs = household_garments(scale=scale)
            plans = [f(gs, w) for f in BASELINES.values()]
            plans.append(solve(gs, w, cfg=SolveConfig(time_limit_s=time_limit)))
            for p in plans:
                assert_valid(p)
                m = p.metrics()
                rows.append({"preset": preset, "scale": scale, "garments": len(gs), **m})
    return pd.DataFrame(rows)


PARETO_CREASE_WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def _pareto(time_limit: float = TIME_LIMIT) -> pd.DataFrame:
    """Sweep the crease weight (reach weight fixed at 1): easy reach vs fewer creases."""
    gs = household_garments(scale=1.5)
    w = load_preset("two_door")
    rows = []
    for lam in PARETO_CREASE_WEIGHTS:
        p = assert_valid(solve(gs, w, cfg=SolveConfig(time_limit_s=time_limit, lam_crease=lam)))
        m = p.metrics()
        modes = pd.Series([p.choice[g].mode for g in p.stored]).value_counts().to_dict()
        rows.append({"lam_crease": lam, **m, **{f"n_{k}": v for k, v in modes.items()}})
    return pd.DataFrame(rows)


def _sensitivity(time_limit: float = TIME_LIMIT) -> pd.DataFrame:
    gs = household_garments(scale=1.5)
    w = load_preset("one_door")
    rows = []
    base = config.k_loft()
    for f in (0.7, 1.0, 1.3):
        with config.override_k_loft(base * f):
            for fn in BASELINES.values():
                m = assert_valid(fn(gs, w)).metrics()
                rows.append({"k_loft_factor": f, "k_loft": round(base * f, 3), **m})
            m = assert_valid(solve(gs, w, cfg=SolveConfig(time_limit_s=time_limit))).metrics()
            rows.append({"k_loft_factor": f, "k_loft": round(base * f, 3), **m})
    return pd.DataFrame(rows)


def _short(method: str) -> str:
    return method.split(" ")[0] if method.startswith("B") else "Optimizer"


def _fig_compare(df: pd.DataFrame, d):
    sub = df[df["scale"] == SCALES[-1]].copy()
    sub["who"] = sub["method"].map(_short)
    metrics = [("stored_value", "Value stored (higher is better)"), ("effort_incl_overflow", "Retrieval effort (lower)"), ("crease_load", "Crease load (lower)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    whos = ["B1", "B2", "B3", "B4", "Optimizer"]
    colours = ["#b0b0b0", "#d9a75f", "#9cc39c", "#8fa9c9", "#1f4e9c"]
    xs = np.arange(len(PRESETS))
    width = 0.16
    for ax, (col, title) in zip(axes, metrics):
        for i, who in enumerate(whos):
            vals = [sub[(sub.preset == p) & (sub.who == who)][col].iloc[0] for p in PRESETS]
            ax.bar(xs + (i - 2) * width, vals, width, label=who, color=colours[i])
        ax.set_xticks(xs, [p.replace("_", "-") for p in PRESETS], fontsize=8)
        ax.set_title(title, fontsize=10)
    labels = ["B1 traditional", "B2 KonMari", "B3 hang-first", "B4 greedy", "Optimizer"]
    fig.legend(labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"E4: optimizer vs baselines, household x{SCALES[-1]:g} ({int(sub['garments'].iloc[0])} garments)", fontsize=11)
    return save_fig(fig, d, "E4_compare.png")


def _fig_pareto(df: pd.DataFrame, d):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.plot(df["crease_load"], df["retrieval_effort"], "o-", color="#1f4e9c")
    for _, r in df.iterrows():
        ax.annotate(f"crease weight {r.lam_crease:g}", (r.crease_load, r.retrieval_effort), fontsize=7, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("Crease load (lower = fewer creases on crease-prone fabric)")
    ax.set_ylabel("Retrieval effort per month (lower = easier)")
    ax.set_title("E4: fewer creases cost easy reach (2-door, household x1.5)", fontsize=9)
    return save_fig(fig, d, "E4_pareto.png")


def run(path=None, time_limit: float = TIME_LIMIT) -> dict:
    d = out_dir(path)
    cmp_df = _compare(time_limit)
    save_csv(cmp_df, d, "E4_compare.csv")
    par_df = _pareto(time_limit)
    save_csv(par_df, d, "E4_pareto.csv")
    sen_df = _sensitivity(time_limit)
    save_csv(sen_df, d, "E4_sensitivity.csv")
    _fig_compare(cmp_df, d)
    _fig_pareto(par_df, d)

    opt = cmp_df[cmp_df.method == "CP-SAT optimizer"].set_index(["preset", "scale"])
    rows = []
    for name in ("B1 traditional flat stacks", "B2 KonMari file fold"):
        b = cmp_df[cmp_df.method == name].set_index(["preset", "scale"])
        rows.append(
            {
                "baseline": name,
                "effort_saving_pct": round(float((100 * (1 - opt.effort_incl_overflow / b.effort_incl_overflow)).mean()), 1),
                "crease_saving_pct": round(float((100 * (1 - opt.crease_load / b.crease_load)).mean()), 1),
                "extra_items_stored": int((opt.stored - b.stored).sum()),
            }
        )
    vs = pd.DataFrame(rows)
    save_csv(vs, d, "E4_savings.csv")
    sens_opt = sen_df[sen_df.method == "CP-SAT optimizer"]
    sens_b1 = sen_df[sen_df.method.str.startswith("B1")]
    robust = bool((sens_opt.objective.values > sens_b1.objective.values).all())
    optimal = int((opt.status == "OPTIMAL").sum())
    summary = {
        "instances": int(len(opt)),
        "optimal_instances": optimal,
        "max_gap": float(opt.gap.max()),
        "savings": vs.to_dict(orient="records"),
        "optimizer_beats_B1_at_all_loft_factors": robust,
    }
    lines = "\n".join(
        f"- vs {r['baseline']}: retrieval effort -{r['effort_saving_pct']}% on average, crease load -{r['crease_saving_pct']}%, {r['extra_items_stored']} more items stored in total"
        for r in summary["savings"]
    )
    write_notes(
        d,
        "E4_notes.md",
        f"""
# E4 - Optimizer vs common practice

{len(PRESETS)} presets x {len(SCALES)} household sizes ({', '.join(f'x{s:g}' for s in SCALES)} of the 3-persona household).
CP-SAT proved optimality on {optimal}/{len(opt)} instances within {TIME_LIMIT:g} s (worst gap {summary['max_gap']:.2e}).
Every plan was re-checked by the independent checker.

{lines}

- Trade-off sweep (E4_pareto.png): raising the crease weight from 0 to 16 moves the plan along a
  curve from easiest reach to fewest creases (more hanging and fewer fold lines); see E4_pareto.csv.
- Loft factor +-30%: optimizer objective beats B1 at every factor: **{robust}**.

Retrieval effort = sum over garments of wears/month x zone cost (1 golden zone, 1.5 bend/stretch,
3 floor/loft) x mode factor (flat stack 1.3, filed/rolled/hung 1.0); garments left out count 5.
These weights are modelling assumptions, stated in fold_and_place/storage/wardrobe.py.
""",
    )
    return summary


if __name__ == "__main__":
    print(run())
