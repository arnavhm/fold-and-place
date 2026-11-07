# E1 - Fold sizes

- **Loft factor** calibrated on T-shirt size M (52 x 71 cm): 18 plies x 0.6 mm
  -> published 0.5 in (1.27 cm) stack gives **k_loft = 1.176** (config holds 1.18).
- **Board fold** (sides in to 23 cm, hem turned up, folded to 30.5 cm) lands within
  0.0% of the 23 x 30.5 cm department-store board for every top tested.
  This checks the geometry (sleeve fold-backs, panel arithmetic); the board width is an input.
- **Equal-panel fold** with the same width misses the board length by
  -22.4% to -12.2%:
  equal thirds come out shorter than a board fold, which tucks the hem first.
- **Area conservation**: worst relative error over 31 categories x 20 random 6-fold
  sequences = 9.05e-16 (numerical noise only).
- **Options**: 18.8 fold options per category on average
  (after dropping options that fit nowhere or are beaten on every dimension).

Limits: geometry only (no cloth physics); single-layer thicknesses are partly estimates
(see configs/fabrics.yaml `basis`); k_loft is calibrated on one published example.
