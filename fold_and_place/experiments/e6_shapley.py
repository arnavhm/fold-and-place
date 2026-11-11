"""E6 - Who should pay for the almirah?

Cost game over the 3-person household (x3 size, so space actually decides the wardrobe):
c(S) = price of the cheapest preset that stores all of S's garments. Compare Shapley shares with
an equal split and splits proportional to garment count and to packed volume; check the core.
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fold_and_place.experiments.common import out_dir, save_csv, save_fig, write_notes
from fold_and_place.folding import generate_options
from fold_and_place.garments import persona_garments
from fold_and_place.mechanisms.shapley import cost_function, in_core, shapley

AGENTS = ["father", "mother", "student"]
SCALE = 3.0


def run(path=None, scale: float = SCALE, time_limit: float = 30.0) -> dict:
    d = out_dir(path)
    c = cost_function(AGENTS, scale, time_limit)
    total = c[frozenset(AGENTS)][0]

    coal = []
    for k in range(1, len(AGENTS) + 1):
        for S in itertools.combinations(AGENTS, k):
            cost, preset = c[frozenset(S)]
            coal.append({"coalition": " + ".join(S), "size": k, "cheapest_wardrobe": preset, "cost_inr": cost})
    save_csv(pd.DataFrame(coal), d, "E6_coalitions.csv")

    owned = {a: persona_garments(a, scale=scale) for a in AGENTS}
    count = {a: len(owned[a]) for a in AGENTS}
    vol = {a: sum(min(o.volume for o in generate_options(g) if o.mode != "hang") if any(o.mode != "hang" for o in generate_options(g)) else 0.0 for g in owned[a]) for a in AGENTS}
    rules = {
        "Shapley value": shapley(AGENTS, c),
        "Equal split": {a: total / len(AGENTS) for a in AGENTS},
        "By garment count": {a: total * count[a] / sum(count.values()) for a in AGENTS},
        "By packed volume": {a: total * vol[a] / sum(vol.values()) for a in AGENTS},
    }
    rows = []
    for name, sh in rules.items():
        ok, bad = in_core(AGENTS, c, sh)
        rows.append({"rule": name, **{a: round(sh[a]) for a in AGENTS}, "in_core": ok, "blocking_groups": "; ".join(bad)})
    shares = pd.DataFrame(rows)
    save_csv(shares, d, "E6_shares.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    x = np.arange(len(rules))
    left = np.zeros(len(rules))
    for k, a in enumerate(AGENTS):
        vals = shares[a].values
        ax.barh(x, vals, left=left, label=a, color=["#8fa9c9", "#d9a75f", "#9cc39c"][k])
        for xi, (lv, v) in enumerate(zip(left, vals)):
            if v > 600:
                ax.text(lv + v / 2, xi, f"₹{v:,.0f}", ha="center", va="center", fontsize=7)
        left += vals
    ax.set_yticks(x, [f"{r}{'' if ok else '  (not in core)'}" for r, ok in zip(shares.rule, shares.in_core)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Share of the ₹{total:,.0f} almirah")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"E6: splitting the cost of one shared {c[frozenset(AGENTS)][1].replace('_', '-')} almirah", fontsize=9)
    save_fig(fig, d, "E6_shares.png")

    shap = rules["Shapley value"]
    summary = {
        "grand_coalition": c[frozenset(AGENTS)][1],
        "cost_inr": total,
        "separate_wardrobes_inr": sum(c[frozenset([a])][0] for a in AGENTS),
        "shapley": {a: round(v) for a, v in shap.items()},
        "core": {r["rule"]: r["in_core"] for r in rows},
    }
    coal_lines = "\n".join(f"| {r['coalition']} | {r['cheapest_wardrobe']} | ₹{r['cost_inr']:,.0f} |" for r in coal)
    share_lines = "\n".join(
        f"| {r['rule']} | " + " | ".join(f"₹{r[a]:,}" for a in AGENTS) + f" | {'yes' if r['in_core'] else 'no: ' + r['blocking_groups']} |"
        for r in rows
    )
    write_notes(
        d,
        "E6_notes.md",
        f"""
# E6 - Who pays for the almirah?

Household x{scale:g} ({', '.join(f'{a} {count[a]}' for a in AGENTS)} garments). Prices are
assumptions (configs/presets.yaml `prices_inr`). Separate wardrobes would cost
₹{summary['separate_wardrobes_inr']:,.0f}; one shared {summary['grand_coalition']} costs ₹{total:,.0f}.

| Group | Cheapest wardrobe that fits all their clothes | Cost |
|---|---|---|
{coal_lines}

| Rule | {' | '.join(AGENTS)} | In the core? |
|---|---|---|---|---|
{share_lines}

Shapley = each person's average extra cost when they join the group, over all joining orders.
In the core = no group would be better off buying its own wardrobe.
""",
    )
    return summary


if __name__ == "__main__":
    print(run())
