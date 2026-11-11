"""Category prompts and the mapping from public dataset labels to our 31 categories."""

from __future__ import annotations

import re

from fold_and_place import config

# Extra phrasings help zero-shot classification (each category's prompts are averaged).
EXTRA_PROMPTS = {
    "saree": ["a saree folded or worn with a pallu over the shoulder", "an Indian sari"],
    "dupatta": ["a long thin dupatta stole", "a chiffon dupatta"],
    "kurti": ["a women's printed kurti", "a long Indian tunic for women"],
    "kurta_men": ["a men's long Indian kurta shirt"],
    "lehenga": ["a flared bridal lehenga skirt with embroidery"],
    "sherwani": ["a long embroidered men's wedding sherwani coat"],
    "dhoti": ["a white unstitched dhoti cloth", "dhoti pants"],
    "lungi": ["a checked cotton lungi wrap"],
    "salwar": ["loose salwar trousers", "churidar leggings"],
    "blouse": ["a short fitted saree blouse choli"],
    "petticoat": ["a plain saree petticoat underskirt"],
    "gown": ["a long flared anarkali gown"],
    "nehru_jacket": ["a sleeveless bandhgala Nehru waistcoat"],
    "tshirt": ["a plain crew-neck t-shirt"],
    "jeans": ["denim jeans laid flat"],
    "innerwear": ["men's briefs or a vest", "women's innerwear"],
    "socks": ["a pair of socks"],
}

PHOTO_TYPE_PROMPTS = {
    "flat": ["a product photo of a garment laid flat on a plain background", "a flat lay photo of clothing"],
    "worn": ["a photo of a model wearing the clothing", "a person wearing an outfit"],
    "folded": ["a photo of a neatly folded garment", "a stack of folded clothes"],
}

FABRIC_PROMPTS = {
    "cotton_woven": ["a woven cotton fabric garment"],
    "cotton_jersey": ["a soft cotton jersey knit garment"],
    "linen": ["a linen garment"],
    "silk": ["a light silk garment"],
    "silk_heavy": ["a heavy silk saree with a zari border"],
    "chiffon": ["a sheer chiffon or georgette garment"],
    "denim": ["a denim garment"],
    "wool_knit_heavy": ["a chunky knitted wool garment"],
    "fleece": ["a fleece sweatshirt fabric"],
    "polyester": ["a shiny polyester garment"],
    "suiting": ["a tailored suiting fabric garment"],
}

EMBELLISH_PROMPTS = {
    "none": ["a plain garment with no embroidery"],
    "light": ["a garment with light embroidery or print"],
    "heavy": ["a heavily embroidered garment with zari, sequins or stone work"],
}

# ---- dataset label maps --------------------------------------------------------------------
INDOFASHION = {
    "saree": "saree",
    "womenkurta": "kurti",
    "leggingsandsalwars": "salwar",
    "palazzos": "palazzo",
    "lehenga": "lehenga",
    "dupattas": "dupatta",
    "blouse": "blouse",
    "gowns": "gown",
    "dhotipants": "dhoti",
    "petticoats": "petticoat",
    "menkurta": "kurta_men",
    "nehrujackets": "nehru_jacket",
    "sherwanis": "sherwani",
    # women_mojari / men_mojari are footwear: not stored in a wardrobe, skipped
}

MYNTRA = {
    "tshirts": "tshirt",
    "shirts": "shirt",
    "jeans": "jeans",
    "trousers": "trousers",
    "shorts": "shorts",
    "trackpants": "trackpants",
    "loungepants": "trackpants",
    "leggings": "leggings",
    "sweatshirts": "hoodie",
    "sweaters": "sweater",
    "jackets": "jacket",
    "blazers": "blazer",
    "dresses": "dress",
    "skirts": "skirt",
    "tops": "top",
    "kurtis": "kurti",
    "sarees": "saree",
    "dupatta": "dupatta",
    "salwar": "salwar",
    "churidar": "salwar",
    "patiala": "salwar",
    "salwarandduppata": "salwar",
    "palazzos": "palazzo",
    "lehengacholi": "lehenga",
    "nehrujackets": "nehru_jacket",
    "waistcoat": "nehru_jacket",
    "socks": "socks",
    "briefs": "innerwear",
    "trunk": "innerwear",
    "boxers": "innerwear",
    "innerwearvests": "innerwear",
    "bra": "innerwear",
}


def norm(label: str) -> str:
    return re.sub(r"[^a-z]", "", str(label).lower())


def map_indofashion(label: str) -> str | None:
    return INDOFASHION.get(norm(label))


def map_myntra(article_type: str, gender: str = "") -> str | None:
    key = norm(article_type)
    if key == "kurtas":
        return "kurta_men" if norm(gender) in ("men", "boys") else "kurti"
    return MYNTRA.get(key)


def category_prompts() -> dict[str, list[str]]:
    out = {}
    for cat, c in config.categories().items():
        out[cat] = [c["prompt"], f"a photo of {c['label'].lower()}"] + EXTRA_PROMPTS.get(cat, [])
    return out
