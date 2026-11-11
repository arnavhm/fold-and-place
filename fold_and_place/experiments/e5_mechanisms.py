"""E5 - Do the mechanisms keep their promises?

On the compact shared section with the 3-person household (scarce space):
1. Run all nine mechanisms on truthful reports: welfare vs optimum, credits paid, budget, fairness.
2. Search for profitable lies (random misreports by one person) - person level, every mechanism.
3. Myerson's own promise, per garment (single-parameter bidders) + a monotonicity check.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fold_and_place.experiments.common import out_dir, save_csv, save_fig, write_notes
from fold_and_place.mechanisms import ALL, Scenario
from fold_and_place.mechanisms.fairness import fairness, proportional_shares
from fold_and_place.mechanisms.mechanisms import Myerson
from fold_and_place.mechanisms.strategic import garment_level_test, misreport_test

# Misreport trials per mechanism: fewer where each trial needs exact re-solves.
TRIALS = {
    "VCG": 60,
    "VCG + Bailey redistribution": 30,
    "Myerson knapsack auction (greedy)": 300,
    "Random serial dictatorship": 15,
    "Maximum Nash welfare": 10,
    "Equal split": 40,
    "First come, first served": 50,
    "Proportional to claims": 40,
    "Priority by claimed value": 300,
}
SHORT = {
    "VCG": "VCG",
    "VCG + Bailey redistribution": "VCG+Bailey",
    "Myerson knapsack auction (greedy)": "Myerson",
    "Random serial dictatorship": "RSD",
    "Maximum Nash welfare": "MNW",
    "Equal split": "Equal",
    "First come, first served": "FCFS",
    "Proportional to claims": "Proportional",
    "Priority by claimed value": "Priority",
}


def run(path=None, scale: float = 0.4, trials_factor: float = 1.0) -> dict:
    d = out_dir(path)
    sc = Scenario.default(scale=scale)
    true = sc.true_values()
    t0 = time.time()
    opt_w = sc.welfare(sc.optimum(true), true)
    ps = proportional_shares(sc, true)

    rows = []
    outcomes = {}
    for m in ALL:
        out = m.run(sc, true)
        outcomes[m.name] = out
        f = fairness(sc, out, true, ps)
        pay = sum(out.payments.values())
        reb = sum(out.rebates.values())
        rows.append(
            {
                "mechanism": m.name,
                "short": SHORT[m.name],
                "uses_money": m.uses_money,
                "welfare": round(out.welfare(sc, true), 2),
                "efficiency_pct": round(100 * out.welfare(sc, true) / opt_w, 1),
                "credits_paid": round(pay, 2),
                "credits_returned": round(reb, 2),
                "net_budget_surplus": round(pay - reb, 2),
                "min_share_ratio": round(f["min_share_ratio"], 3),
                "below_proportional_share": f["below_proportional_share"],
                "max_envy_points": round(f["max_envy"], 2),
                "ef1_violations": round(f["ef1_violations"], 3),
                "nash_product": round(f["nash_product"], 0),
                **{f"value_{a}": round(f["stored_value"][a], 1) for a in sc.agents},
                **{f"utility_{a}": round(out.utility(sc, a, true), 2) for a in sc.agents},
                **{f"pays_{a}": round(out.payments.get(a, 0.0) - out.rebates.get(a, 0.0), 2) for a in sc.agents},
                "claim": m.claim,
            }
        )
    res = pd.DataFrame(rows)
    save_csv(res, d, "E5_outcomes.csv")

    lies = []
    for m in ALL:
        n = max(3, int(round(TRIALS[m.name] * trials_factor)))
        lies.append(misreport_test(m, sc, n, seed=11))
    g = garment_level_test(Myerson(), sc, max(30, int(300 * trials_factor)), seed=12)
    lie_df = pd.DataFrame(lies + [g])
    save_csv(lie_df.drop(columns=["example"], errors="ignore"), d, "E5_misreports.csv")

    # property matrix
    prop = res[["short", "efficiency_pct", "net_budget_surplus", "below_proportional_share", "max_envy_points", "ef1_violations"]].copy()
    prop["profitable_lie_rate"] = [lies[i]["rate"] for i in range(len(ALL))]
    save_csv(prop, d, "E5_property_matrix.csv")
    _figure(res, prop, d, sc)

    summary = {
        "garments": len(sc.garments),
        "optimal_welfare": round(opt_w, 2),
        "solves": sc.solves,
        "seconds": round(time.time() - t0, 1),
        "truthful_found": {r["mechanism"]: f"{r['profitable']}/{r['trials']}" for r in lies + [g]},
        "myerson_non_monotone": g["non_monotone"],
    }
    _notes(d, res, lie_df, g, summary, sc)
    return summary


def _figure(res, prop, d, sc):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    y = np.arange(len(res))
    labels = res["short"].tolist()
    colours = ["#1f4e9c" if m else "#7fa37f" for m in res["uses_money"]]
    axes[0].barh(y, res["efficiency_pct"], color=colours)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Welfare, % of optimum (blue = uses credits)", fontsize=9)
    axes[0].set_xlim(0, 105)
    axes[1].barh(y, prop["profitable_lie_rate"] * 100, color="#c0504d")
    axes[1].set_yticks(y, labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title("Random lies that paid off, %", fontsize=9)
    stored = res[[f"value_{a}" for a in sc.agents]].values
    left = np.zeros(len(res))
    for k, a in enumerate(sc.agents):
        axes[2].barh(y, stored[:, k], left=left, label=a, color=["#8fa9c9", "#d9a75f", "#9cc39c"][k % 3])
        left += stored[:, k]
    axes[2].set_yticks(y, labels, fontsize=8)
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].set_title("Value stored per person", fontsize=9)
    fig.suptitle("E5: nine ways to share one wardrobe (compact section, 3 people, truthful reports)", fontsize=10)
    return save_fig(fig, d, "E5_mechanisms.png")


def _notes(d, res, lie_df, g, summary, sc):
    lines = []
    for _, r in res.iterrows():
        lines.append(
            f"| {r['short']} | {r['efficiency_pct']}% | {r['net_budget_surplus']} | {r['below_proportional_share']} | "
            f"{r['max_envy_points']} | {r['ef1_violations']} |"
        )
    lie_lines = [f"| {r['mechanism']} | {r['profitable']}/{r['trials']} | {r['max_gain_points']} |" for _, r in lie_df.iterrows()]
    write_notes(
        d,
        "E5_notes.md",
        f"""
# E5 - Mechanism properties

Scenario: compact shared section, {summary['garments']} garments, {len(sc.agents)} people; optimal welfare
{summary['optimal_welfare']} points. Exact CP-SAT solves with a value-independent option set
(warm_start off), so VCG's truthfulness argument applies. {summary['solves']} solves, {summary['seconds']} s.

| Mechanism | Welfare vs optimum | Net credits kept | People below proportional share | Max envy (points) | EF1 violations |
|---|---|---|---|---|---|
{chr(10).join(lines)}

Random lies by one person (others truthful):

| Mechanism | Profitable lies found | Largest gain (points) |
|---|---|---|
{chr(10).join(lie_lines)}

Myerson per garment: {g['profitable']}/{g['trials']} profitable, {g['non_monotone']} monotonicity violations.
A person owning several garments CAN gain by misreporting under Myerson (one garment's report moves
their other garments); its guarantee is per single-parameter bidder (garment), not per person.

MNW is NOT strategyproof in theory; finding no profitable lie in a few trials on this instance does
not change that. Equal split and FCFS are truthful only because they ignore values (and waste space).

Fairness is measured on SPACE: envy_ij = most value i could store in j's space minus in i's own
(volume/rod knapsack, optimistic for everyone). EF1 = envy disappears after removing j's largest item.
Trial counts are small for solver-heavy mechanisms; "0 found" is evidence, not proof - the proofs are
VCG (Clarke), Myerson's lemma and serial dictatorship.
""",
    )


if __name__ == "__main__":
    print(run())
