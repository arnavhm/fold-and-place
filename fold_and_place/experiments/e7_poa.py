"""E7 - What does lying cost under naive rules?

Under the manipulable mechanisms (priority by claimed value, space proportional to claims), each
person repeatedly picks the report multiplier (0.5x .. 3x of their true values) that is best for
them given the others. Where this settles is an equilibrium of the restricted game; its TRUE
welfare divided by the optimum is an empirical price of anarchy. Truthful mechanisms (VCG, RSD)
are run the same way as a control: nobody should move.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fold_and_place.experiments.common import out_dir, save_csv, save_fig, write_notes
from fold_and_place.mechanisms import Scenario
from fold_and_place.mechanisms.mechanisms import RSD, VCG, Priority, Proportional
from fold_and_place.mechanisms.strategic import best_response_dynamics


def run(path=None, scale: float = 0.4) -> dict:
    d = out_dir(path)
    sc = Scenario.default(scale=scale)
    true = sc.true_values()
    opt = sc.welfare(sc.optimum(true), true)
    rows, hist_rows = [], []
    for mech in (Priority(), Proportional(), VCG(), RSD()):
        truthful = mech.run(sc, true).welfare(sc, true)
        br = best_response_dynamics(mech, sc, rounds=8)
        eq = br["outcome"].welfare(sc, true)
        rows.append(
            {
                "mechanism": mech.name,
                "truthful_welfare": round(truthful, 2),
                "equilibrium_welfare": round(eq, 2),
                "optimal_welfare": round(opt, 2),
                "poa_ratio": round(eq / opt, 3),
                "loss_vs_truthful_pct": round(100 * (1 - eq / truthful), 1) if truthful else 0.0,
                "converged": br["converged"],
                "rounds": br["rounds"],
                **{f"report_x_{a}": br["multipliers"][a] for a in sc.agents},
            }
        )
        for t, h in enumerate(br["history"]):
            hist_rows.append({"mechanism": mech.name, "round": t, **{a: h[a] for a in sc.agents}})
    df = pd.DataFrame(rows)
    save_csv(df, d, "E7_poa.csv")
    save_csv(pd.DataFrame(hist_rows), d, "E7_dynamics.csv")

    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    x = np.arange(len(df))
    ax.bar(x - 0.18, df.truthful_welfare / opt * 100, 0.36, label="everyone truthful", color="#8fa9c9")
    ax.bar(x + 0.18, df.equilibrium_welfare / opt * 100, 0.36, label="after best responses", color="#1f4e9c")
    ax.set_xticks(x, ["Priority", "Proportional", "VCG", "RSD"], fontsize=8)
    ax.set_ylabel("Welfare, % of optimum")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.set_title("E7: manipulable rules lose welfare when people game them; truthful ones do not move", fontsize=9)
    save_fig(fig, d, "E7_poa.png")

    lines = "\n".join(
        f"| {r['mechanism']} | {r['truthful_welfare']} | {r['equilibrium_welfare']} | {r['poa_ratio']} | "
        + ", ".join(f"{a} x{r[f'report_x_{a}']:g}" for a in sc.agents)
        + f" | {'yes' if r['converged'] else 'no'} |"
        for r in rows
    )
    write_notes(
        d,
        "E7_notes.md",
        f"""
# E7 - Price of anarchy (empirical)

Compact shared section, optimal welfare {opt:.2f} points. Strategy space: one report multiplier per
person in {{0.5, 1, 1.5, 2, 3}} x true values; best-response dynamics from truthful reports.

| Mechanism | Welfare if truthful | Welfare at equilibrium | Equilibrium / optimum | Reports at equilibrium | Converged |
|---|---|---|---|---|---|
{lines}

The ratio is an estimate for this instance and this restricted strategy space, not a worst-case bound.
""",
    )
    return {r["mechanism"]: {"poa": r["poa_ratio"], "reports": {a: r[f"report_x_{a}"] for a in sc.agents}} for r in rows}


if __name__ == "__main__":
    print(run())
