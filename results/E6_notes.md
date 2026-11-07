# E6 - Who pays for the almirah?

Household x3 (father 84, mother 105, student 102 garments). Prices are
assumptions (configs/presets.yaml `prices_inr`). Separate wardrobes would cost
₹34,000; one shared two_door costs ₹16,000.

| Group | Cheapest wardrobe that fits all their clothes | Cost |
|---|---|---|
| father | one_door | ₹9,000 |
| mother | two_door | ₹16,000 |
| student | one_door | ₹9,000 |
| father + mother | two_door | ₹16,000 |
| father + student | one_door | ₹9,000 |
| mother + student | two_door | ₹16,000 |
| father + mother + student | two_door | ₹16,000 |

| Rule | father | mother | student | In the core? |
|---|---|---|---|---|
| Shapley value | ₹3,000 | ₹10,000 | ₹3,000 | yes |
| Equal split | ₹5,333 | ₹5,333 | ₹5,333 | no: father + student |
| By garment count | ₹4,619 | ₹5,773 | ₹5,608 | no: father + student |
| By packed volume | ₹3,431 | ₹6,824 | ₹5,746 | no: father + student |

Shapley = each person's average extra cost when they join the group, over all joining orders.
In the core = no group would be better off buying its own wardrobe.
