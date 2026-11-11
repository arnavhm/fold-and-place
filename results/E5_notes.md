# E5 - Mechanism properties

Scenario: compact shared section, 36 garments, 3 people; optimal welfare
172.01 points. Exact CP-SAT solves with a value-independent option set
(warm_start off), so VCG's truthfulness argument applies. 359 solves, 74.2 s.

| Mechanism | Welfare vs optimum | Net credits kept | People below proportional share | Max envy (points) | EF1 violations |
|---|---|---|---|---|---|
| VCG | 100.0% | 23.4 | 0 | 4.0 | 1.0 |
| VCG+Bailey | 100.0% | 20.09 | 0 | 4.0 | 1.0 |
| Myerson | 72.0% | 72.3 | 0 | 21.0 | 2.0 |
| RSD | 82.0% | 0.0 | 0 | 45.0 | 2.167 |
| MNW | 99.5% | 0.0 | 0 | 4.0 | 0.0 |
| Equal | 61.0% | 0.0 | 0 | 13.0 | 0.0 |
| FCFS | 73.7% | 0.0 | 0 | 7.0 | 0.0 |
| Proportional | 64.0% | 0.0 | 0 | 15.0 | 1.0 |
| Priority | 72.0% | 0.0 | 0 | 21.0 | 2.0 |

Random lies by one person (others truthful):

| Mechanism | Profitable lies found | Largest gain (points) |
|---|---|---|
| VCG | 0/60 | 0.0 |
| VCG + Bailey redistribution | 0/30 | 0.0 |
| Myerson knapsack auction (greedy) | 78/300 | 21.218 |
| Random serial dictatorship | 0/15 | 0.0 |
| Maximum Nash welfare | 0/10 | 0.0 |
| Equal split | 0/40 | 0.0 |
| First come, first served | 0/50 | 0.0 |
| Proportional to claims | 18/40 | 12.085 |
| Priority by claimed value | 160/300 | 24.332 |
| Myerson knapsack auction (greedy) [per garment] | 0/300 | 0.0 |

Myerson per garment: 0/300 profitable, 0 monotonicity violations.
A person owning several garments CAN gain by misreporting under Myerson (one garment's report moves
their other garments); its guarantee is per single-parameter bidder (garment), not per person.

MNW is NOT strategyproof in theory; finding no profitable lie in a few trials on this instance does
not change that. Equal split and FCFS are truthful only because they ignore values (and waste space).

Fairness is measured on SPACE: envy_ij = most value i could store in j's space minus in i's own
(volume/rod knapsack, optimistic for everyone). EF1 = envy disappears after removing j's largest item.
Trial counts are small for solver-heavy mechanisms; "0 found" is evidence, not proof - the proofs are
VCG (Clarke), Myerson's lemma and serial dictatorship.
