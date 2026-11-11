# E4 - Optimizer vs common practice

4 presets x 2 household sizes (x1, x2 of the 3-persona household).
CP-SAT proved optimality on 6/8 instances within 30 s (worst gap 1.90e-04).
Every plan was re-checked by the independent checker.

- vs B1 traditional flat stacks: retrieval effort -24.4% on average, crease load -63.6%, 23 more items stored in total
- vs B2 KonMari file fold: retrieval effort -47.2% on average, crease load -64.7%, 6 more items stored in total

- Trade-off sweep (E4_pareto.png): raising the crease weight from 0 to 16 moves the plan along a
  curve from easiest reach to fewest creases (more hanging and fewer fold lines); see E4_pareto.csv.
- Loft factor +-30%: optimizer objective beats B1 at every factor: **True**.

Retrieval effort = sum over garments of wears/month x zone cost (1 golden zone, 1.5 bend/stretch,
3 floor/loft) x mode factor (flat stack 1.3, filed/rolled/hung 1.0); garments left out count 5.
These weights are modelling assumptions, stated in fold_and_place/storage/wardrobe.py.
