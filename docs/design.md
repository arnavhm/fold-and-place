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
| Mask | rembg U²-Net (flat lay), SegFormer-B2 clothes (worn), plain-background fallback (border colour model, Otsu threshold bounded by the sheet's own texture); holes filled, specks and the reference sheet removed, main part kept | |
| Reference | **found automatically** (`vision.reference`): a printed ArUco marker, or any **A4 sheet** - bright, low-saturation convex quads (and edge loops for white paper on pale sheets) with the A4 side ratio and near-right corners, scored by size and whiteness, corners refined to sub-pixel | a sheet on the floor also means "flat-lay photo" |
| Upright | `vision.shape.orient`: the smallest turn that squares the garment's minimum-area rectangle, then the quarter turn that maximises left-right mirror symmetry plus a cue per template (armpit step near the top for tops; two legs at the bottom for trousers; wider hem for skirts; length vertical for rectangles) | photos at any angle |
| Size | **with a reference**: homography from 4 corners (numpy DLT) → top-down cm grid → template-aware widths/lengths (tops: body above the hem; dresses/kurtis: chest just below the armpit; trousers/skirts: waistband); **without**: the photo's own proportions scaled to the size label's chart length (`size_source="approx"`); worn photos: size chart; drapes: fabric standard | every record carries `size_source` and a confidence |
| Outline | the upright mask, symmetrised about its centre line, as a polygon in cm (`Garment.outline`) | the fold engine folds *this* garment |

Without the recognition models the app still measures (`pipeline.measure_photo`): the person says
what the garment is, the photo gives its size and outline.

E2 measures classification accuracy on held-out public images. E3 measures size error: on 144
synthetic photos with a perfect mask (geometry only), and on 54 **realistic** synthetic photos
(`vision.synth`: textured bedsheet, prints, creases, shadows, uneven light, any rotation, tilt up
to 25°, JPEG) through the whole chain with the sheet found automatically - sheet found in all 54,
mean error 1.5 cm (width) and 1.8 cm (length), median 1 cm. E8 repeats it on real phone photos.

## 3. Fold: geometry (`folding/`)

A garment is a polygon - its own photographed outline when it came from a photo, else a template (`templates.outline`: tee, dress, skirt, trousers, rectangle) - with a ply
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

**House rules** (`storage.rules`) restrict where a garment may go - an `allowed` map that removes
(option, compartment) pairs before solving: personal shelves (a person's clothes only in their own
or shared compartments), packing winter clothes away (loft or outside everyone's easy-reach band)
and keeping heavy things (denim, heavy wool, suiting, heavy embroidery) out of the loft. Each
person's easy-reach band runs from 55 cm to their shoulder height (0.82 × their height).

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

**Payments.** In a household nobody pays rupees for shelf space, so VCG payments are read as
**chore credits**: one credit is one point of value, settled in chores. A person pays exactly the
value their clothes push out of the wardrobe for everyone else.

**Determinism.** Several plans can tie on welfare. The mechanism scenario runs the solver with one
worker and a fixed seed, so the same plan - and the same payments - comes out on every run.

**Fairness** is measured on space: envy_ij = most value *i* could store in *j*'s space minus in
its own (volume/rod knapsack); EF1 = envy disappears after removing *j*'s largest item;
proportional share = what *i* stores under equal split.

**Strategic tests** (`strategic.py`): random lies by one person (scale all values, scale each,
or change one), checking whether the liar's *true* utility rises; garment-level tests for
Myerson; best-response dynamics over report multipliers {0.5, 1, 1.5, 2, 3} for an empirical
price of anarchy.

**Collusion** (`strategic.collusion_test`): two people agree a joint lie (each scales all their
values by ×0.5-×3) and we check whether their *combined* true utility rises. Truthfulness for one
person does not imply it for groups. On the default scenario VCG is open to it - pairs gain 3-6
points by jointly *understating*, which cuts what each pays for the other - while random serial
dictatorship gains at most 0.8 for one pair (through the small reach terms) and nothing for the others.

Results (E5, 36 garments in a compact shared section): VCG 100% efficient with 0/60 profitable
lies; RSD 82% and 0/15; MNW 99.9%; Priority 72% with 160/300 profitable lies; Proportional 64% with
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
* Several optimal plans can tie. The household planner uses parallel workers, so among equally good
  plans the one shown can vary between runs; the mechanism scenario uses one worker, so its plans
  and payments are reproducible.
* The approximate size (no reference sheet) cannot undo camera tilt; it assumes a photo from above
  and a correct size label.

## 8. Dashboard

A guided flow in Streamlit (`app/app.py`, pages in `app/views/`): **Start → 1 Your wardrobe →
2 Your clothes → 3 Your plan → 4 Sharing fairly**, with *Fold any garment*, *Experiments* and
*How it works* behind the scenes. Each screen has one headline sentence, one main visual and at most
four plain-language numbers; the detail sits in "Show the maths". The wardrobe is drawn as SVG with
every garment in its owner's colour; the plan opens the 3D fold of any garment in a dialog; the
sharing page tells the exaggeration story with one verdict card per rule. Households can be saved to
and loaded from JSON, and the plan can be downloaded as a printable shelf-by-shelf sheet.
