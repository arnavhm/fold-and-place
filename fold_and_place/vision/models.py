"""Model wrappers. Heavy libraries are imported lazily, so the rest of the project (and CI) never
needs torch, open_clip, transformers or rembg. Weights download on first use (Hugging Face cache)
and are never committed.

  Embedder        Marqo-FashionSigLIP (Apache-2.0), fashion-tuned SigLIP image/text embeddings
  FlatSegmenter   rembg (MIT) background removal, for flat-lay / product photos
  WornSegmenter   SegFormer-B2 clothes parser (mattmdjaga/segformer_b2_clothes), for worn photos
  PlainSegmenter  numpy fallback: pixels that differ from the border colour (plain backgrounds)
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

FASHION_SIGLIP = "hf-hub:Marqo/marqo-fashionSigLIP"
SEGFORMER_CLOTHES = "mattmdjaga/segformer_b2_clothes"
# SegFormer clothes labels that count as "the garment"
SEGFORMER_GARMENT_LABELS = {4: "upper", 5: "skirt", 6: "pants", 7: "dress", 17: "scarf"}


def _device() -> str:  # pragma: no cover - hardware dependent
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:  # pragma: no cover - needs model weights
    def __init__(self, name: str = FASHION_SIGLIP):
        self.name = name

    @cached_property
    def _model(self):
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(self.name)
        tokenizer = open_clip.get_tokenizer(self.name)
        dev = _device()
        model = model.to(dev).eval()
        return model, preprocess, tokenizer, dev, torch

    def images(self, imgs, batch: int = 32) -> np.ndarray:
        model, preprocess, _, dev, torch = self._model
        out = []
        for i in range(0, len(imgs), batch):
            x = torch.stack([preprocess(im.convert("RGB")) for im in imgs[i : i + batch]]).to(dev)
            with torch.no_grad():
                e = model.encode_image(x, normalize=True)
            out.append(e.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def texts(self, texts: list[str]) -> np.ndarray:
        model, _, tok, dev, torch = self._model
        with torch.no_grad():
            e = model.encode_text(tok(texts).to(dev), normalize=True)
        return e.float().cpu().numpy()

    def prompt_bank(self, prompts: dict[str, list[str]]) -> tuple[list[str], np.ndarray]:
        """One normalised embedding per label: the mean of its prompts."""
        labels, vecs = [], []
        for lab, ps in prompts.items():
            v = self.texts(ps).mean(axis=0)
            vecs.append(v / np.linalg.norm(v))
            labels.append(lab)
        return labels, np.stack(vecs)


class FlatSegmenter:  # pragma: no cover - needs model weights
    def __init__(self, model: str = "u2net"):
        self.model = model

    @cached_property
    def _session(self):
        from rembg import new_session

        return new_session(self.model)

    def __call__(self, img) -> np.ndarray:
        from rembg import remove

        m = remove(img.convert("RGB"), session=self._session, only_mask=True)
        return np.asarray(m) > 127


class WornSegmenter:  # pragma: no cover - needs model weights
    @cached_property
    def _model(self):
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

        proc = SegformerImageProcessor.from_pretrained(SEGFORMER_CLOTHES)
        model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER_CLOTHES).eval()
        return proc, model

    def __call__(self, img) -> np.ndarray:
        import torch

        proc, model = self._model
        with torch.no_grad():
            logits = model(**proc(images=img.convert("RGB"), return_tensors="pt")).logits
        up = torch.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
        seg = up.argmax(dim=1)[0].numpy()
        return np.isin(seg, list(SEGFORMER_GARMENT_LABELS))


class PlainSegmenter:
    """Fallback for plain backgrounds: everything that differs from the border colour."""

    def __init__(self, threshold: float = 40.0):
        self.threshold = threshold

    def __call__(self, img) -> np.ndarray:
        a = np.asarray(img.convert("RGB"), dtype=float)
        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        bg = np.median(border, axis=0)
        return np.linalg.norm(a - bg, axis=2) > self.threshold


def softmax(z: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = np.asarray(z, float) * temp
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def zero_shot(img_emb: np.ndarray, label_emb: np.ndarray, temp: float = 100.0) -> np.ndarray:
    """Class probabilities from cosine similarity (rows: images, cols: labels)."""
    return softmax(np.atleast_2d(img_emb) @ label_emb.T, temp)


class LinearHead:
    """Multinomial logistic regression on frozen embeddings. Trained with scikit-learn, stored as
    plain numpy (models/head.npz) so inference needs nothing but numpy."""

    def __init__(self, classes: list[str], coef: np.ndarray, intercept: np.ndarray):
        self.classes, self.coef, self.intercept = list(classes), coef, intercept

    @classmethod
    def train(cls, X: np.ndarray, y: list[str], C: float = 2.0) -> LinearHead:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(X, y)
        coef, inter = clf.coef_, clf.intercept_
        if len(clf.classes_) == 2:  # sklearn stores binary problems as one row
            coef, inter = np.vstack([-coef, coef]), np.array([-inter[0], inter[0]])
        return cls(list(clf.classes_), coef, inter)

    def proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(np.atleast_2d(X) @ self.coef.T + self.intercept)

    def save(self, path) -> None:
        np.savez(path, classes=np.array(self.classes), coef=self.coef, intercept=self.intercept)

    @classmethod
    def load(cls, path) -> LinearHead:
        d = np.load(path, allow_pickle=False)
        return cls([str(c) for c in d["classes"]], d["coef"], d["intercept"])


def ensemble(zs: np.ndarray, zs_labels: list[str], head: LinearHead | None, head_p: np.ndarray | None) -> np.ndarray:
    """Average the zero-shot and trained-head probabilities on the full label set. Categories the
    head never saw (no public training data) keep their zero-shot probability."""
    if head is None or head_p is None:
        return zs
    full = np.zeros_like(zs)
    idx = {lab: i for i, lab in enumerate(zs_labels)}
    for j, lab in enumerate(head.classes):
        if lab in idx:
            full[:, idx[lab]] = head_p[:, j]
    return 0.5 * zs + 0.5 * full
