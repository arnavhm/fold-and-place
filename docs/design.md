# Design

This document describes what the system computes and why, at the level a report needs. Code
references are given as `module.function`.

## 1. Problem

A household of *n* people shares one wardrobe. Each person owns garments; each garment *g* has an
owner *o(g)*, a category, a flat size, a fabric, a wear rate *f_g* (wears per month) and a value
*θ_g* (importance, 1-10) that only its owner knows. We want to:

1. **See** - turn a photo into a garment record (category, fabric, flat size).
2. **Fold** - list every sensible way to fold it and the packet each produces.
3. **Place** - choose one fold per garment and a place for it, using the space well, putting
   frequently worn things in easy reach and avoiding creases on delicate fabrics.
4. **Share** - when not everything fits, decide whose garments get the space from *reported*
   values, in a way that is efficient, fair, and hard to game.

## 2. See: photo → garment record (`vision/`)

| Step | Method | Notes |
|---|---|---|
| Embedding | Marqo-FashionSigLIP (`vision.models.Embedder`) | fashion-tuned SigLIP; image + text in one space |
| Category | zero-shot prompts for 31 categories, ensembled with a logistic-regression head trained on IndoFashion + Myntra embeddings (`LinearHead`, `ensemble`) | head is optional; zero-shot alone works |
| Photo type, fabric, embellishment | zero-shot prompts | heavy embroidery is assumed for lehenga/sherwani unless clearly plain |
| Mask | rembg U²-Net (flat lay), SegFormer-B2 clothes (worn), plain-background threshold fallback | |
| Size | **with a reference** (A4 sheet or ArUco marker): homography from 4 corners (numpy DLT), rectified mask, template-aware widths/lengths (`measure.measure_with_marker`); **without**: the category's size chart for the labelled size | every record carries `size_source` and a confidence |

E2 measures classification accuracy on held-out public images; E3 measures size error on 144
synthetic photos with tilt and corner noise (mean 2.3 cm width, 1.9 cm length).

## 3. Fold: geometry (`folding/`)

A garment is a polygon (`templates.outline`: tee, dress, skirt, trousers, rectangle) with a ply
count (2 for front + back, 1 for a saree). A fold is `FoldStep(axis, coord, move)`: the part on one
side of an axis-aligned line is reflected across it (`engine.apply_fold`, shapely). Area × plies is
conserved exactly (E1: worst error 9e-16).

For each packet: footprint *a × b* from the polygon bounds, the maximum number of plies *k* by
grid sampling, and height *h = k · t_fabric · k_loft*. The loft factor *k_loft = 1.18* is
calibrated on the published department-store tee fold (11 × 9 × 0.5 in).

Fold families (`options.generate_options`): sides-in then halves/thirds/quarters along the length,
the department-store board fold (23 × 30.5 cm), leg-over-leg for trousers, width-then-accordion
panels for sarees, dupattas and dhotis, rolls, and hanging (with rod width and drop). Each is
offered in **flat** (stacked) and **file** (on edge, KonMari) modes. Options that fit no
compartment of any preset, or are no better than another option on every dimension, are dropped;
named methods are always kept. About 19 options per category remain.

**Animation** (`app/fold3d.py`): the outline becomes a mesh whose grid includes every fold line's
pre-image on the flat garment, so the cloth bends exactly on the crease. A turned-over stack lands
on what lies under it: *z → base + top − z*, which is a 180° rotation about a hinge at height
*(base + top)/2*; the browser rotates by θ ∈ [0, π].

## 4. Place: exact packing (`storage/model.py`)

Compartments: shelves and lofts hold side-by-side **stacks**; drawers hold front-to-back **rows**
of filed packets; rails hold hangers. Decision: for every garment at most one (fold option, lane).

Constraints (integer millimetres; items rounded up, capacities down):

* stack: Σ lane widths ≤ shelf width; in each lane every packet's width ≤ lane width and
  Σ heights ≤ clearance; packet length ≤ depth;
* row: Σ row depths ≤ drawer depth; Σ thicknesses in a row ≤ drawer width; packet height ≤
  drawer height;
* rail: Σ rod widths ≤ rail length; drop ≤ rail height;
* embroidered pieces fold flat only and nothing goes on top of them.

Objective (maximised):

  Σ_stored 20·θ_g + λ_r·f_g·(5 − reach_g) − λ_c·crease_g·creases_g

reach = zone cost (1 inside the owner's golden zone 55 cm – shoulder height, 1.5 just outside,
3 far outside or loft) × mode factor (flat stack 1.3, otherwise 1); a garment left out counts as
reach 5 (a chair pile). Value dominates; then effort; then creases.

Solved with OR-Tools CP-SAT, warm-started from the best baseline. Per compartment, options that are
no better on (lane space, in-lane space, creases) are pruned - lossless, because reach is the same
for every option in one compartment. An independent checker re-verifies every plan.

Baselines: B1 traditional flat stacks, B2 KonMari filing, B3 hang everything possible, B4 greedy
by value density. E4: across 4 presets × 2 household sizes, CP-SAT is optimal on 6/8 (gap ≤ 2e-4)
and cuts reach effort 24% and crease load 64% against B1 while storing 23 more items.

## 5. Share: mechanisms (`mechanisms/`)

**Model.** Agents *N* (|N| = 3). Agent *i*'s type is θ_i = (θ_g) for the garments it owns. An
outcome *x* is a plan. Agent *i*'s value is

  v_i(x; θ_i) = Σ_{g ∈ stored(x), o(g)=i} ( θ_g + e_g(x) ),

where e_g(x) = [λ_r f_g (5 − reach_g) − λ_c crease_g creases_g] / 20 is the small,
publicly known reach/crease term. Utility is quasi-linear: u_i = v_i − p_i (+ rebate).
Reports are rounded to half points. The set of feasible plans does not depend on reports (the
option set is value-independent), which VCG's argument needs.

| Mechanism | Allocation | Payment | Guarantee |
|---|---|---|---|
| **VCG** | x* = argmax_x Σ_i v_i(x; r_i) (exact CP-SAT) | p_i = max_x Σ_{j≠i} v_j(x; r_j) − Σ_{j≠i} v_j(x*; r_j), the first term computed with *i*'s garments absent (Clarke pivot) | dominant-strategy truthful, efficient, individually rational, no deficit |
| **VCG + Bailey** | as VCG | VCG payment minus rebate (1/n)·Rev(VCG without *i*) | same, smaller surplus |
| **Myerson** | each garment is a single-parameter bidder; greedy by reported value then density (monotone) | each stored garment pays its critical value (bisection) | truthful per garment; not per person with several garments |
| **RSD** | uniform random order; each person in turn maximises own value given what earlier people secured (exact sequential solves, all *n!* orders) | none | strategyproof, ex-post Pareto, equal treatment |
| **MNW** | argmax Π_i stored value_i | none | fair (EF1 when goods are unconstrained); not strategyproof |
| Equal split | each person gets 1/n of every compartment | none | ignores values |
| FCFS | fixed arrival order | none | ignores values |
| Proportional | space split by share of claimed value | none | manipulable |
| Priority | greedy by claimed value | none | manipulable |

**Fairness** is measured on space: envy_ij = most value *i* could store in *j*'s space minus in
its own (volume/rod knapsack); EF1 = envy disappears after removing *j*'s largest item;
proportional share = what *i* stores under equal split.

**Strategic tests** (`strategic.py`): random lies by one person (scale all values, scale each,
or change one), checking whether the liar's *true* utility rises; garment-level tests for
Myerson; best-response dynamics over report multipliers {0.5, 1, 1.5, 2, 3} for an empirical
price of anarchy.

Results (E5, 36 garments in a compact shared section): VCG 100% efficient with 0/60 profitable
lies; RSD 82% and 0/15; MNW 99.5%; Priority 72% with 160/300 profitable lies; Proportional 64% with
18/40. E7: Priority's equilibrium reaches 0.81 of optimum, Proportional 0.64 (everyone claims ×3).

## 6. Who pays for the wardrobe (`mechanisms/shapley.py`)

Cost game: c(S) = price of the cheapest preset that fits every garment of coalition *S*
(exact feasibility solves). Shapley value φ_i = average marginal cost of *i* over join orders.
The core: Σ_{i∈S} x_i ≤ c(S) for all S. E6 (household × 3): one shared two-door (₹16,000)
instead of separate wardrobes (₹34,000); Shapley ₹3,000 / ₹10,000 / ₹3,000 is in the core, while
equal split and split-by-count are blocked by father + student, who fit in a one-door together.

## 7. Assumptions and limits

* Household composition, values and wear rates are labelled assumptions (`configs/personas.yaml`).
* Fabric thicknesses are partly estimates (`configs/fabrics.yaml`, column `basis`); *k_loft* is
  calibrated on one published example.
* Folding is geometry, not cloth physics; the animation's heights are a sketch.
* Effort weights (zone costs, mode factor, chair pile = 5) are modelling choices; E4 reports a
  sensitivity sweep.
* "0 profitable lies found" is evidence, not proof; the proofs are Clarke's (VCG), Myerson's
  lemma and serial dictatorship.
* Several optimal plans can tie; with parallel CP-SAT workers the chosen one may vary between
  runs. VCG payments can then shift between people, but every person's utility (total welfare
  minus the others' best welfare without them) is unchanged.
