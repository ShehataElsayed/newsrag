"""Experimental optional neural relevance ranker. No pretrained weights bundled.

Requires NumPy only when used. Training labels must be independently reviewed.
Neural ranking is not claim verification. An abstention gate is separate from
NewsroomRAG's permutation-only reranker and does not change its public API.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import Embedder, Evidence, ValidationError, _terms


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise ImportError('Install the experimental extra: newsrag[neural]') from exc
    return numpy


@dataclass(frozen=True)
class ReviewedPair:
    """Relative passage relevance, not a true/false claim label."""
    query: str
    positive: str
    negative: str
    event_group: str
    review_status: str

    def __post_init__(self) -> None:
        if self.review_status != 'independently_reviewed':
            raise ValidationError('Training pairs require independent relevance review')
        if any(not isinstance(x, str) or not x.strip()
               for x in (self.query, self.positive, self.negative, self.event_group)):
            raise ValidationError('Nonempty query, passages and event group required')
        if self.positive == self.negative:
            raise ValidationError('Positive and negative passages must differ')


class PairFeatures:
    """Frozen embeddings plus six observed relevance features, no truth score."""
    def __init__(self, embed: Embedder, *, dimension: int, encoder_id: str,
                 encoder_revision: str):
        if not isinstance(dimension, int) or dimension < 1:
            raise ValidationError('Positive embedding dimension required')
        if not encoder_id.strip() or not encoder_revision.strip():
            raise ValidationError('Explicit encoder ID and revision required')
        self.embed = embed
        self.dimension = dimension
        self.encoder_id = encoder_id
        self.encoder_revision = encoder_revision

    @property
    def width(self) -> int:
        """Number of feature columns produced per (query, text) pair."""
        return 4 * self.dimension + 6

    def transform(self, pairs: Sequence[tuple[str, str]]) -> Any:
        """Map (query, text) pairs to hashed pairwise feature rows."""
        np = _numpy()
        if not pairs:
            return np.empty((0, self.width), dtype=np.float64)
        texts = [text for pair in pairs for text in pair]
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValidationError('Nonempty input texts required')
        vectors = np.asarray(self.embed(texts), dtype=np.float64)
        if vectors.shape != (len(texts), self.dimension) or not np.isfinite(vectors).all():
            raise ValidationError('Encoder returned inconsistent or nonfinite vectors')
        # No corpus-relative normalization that inflates weak singleton matches.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        output = []
        for i, (query, passage) in enumerate(pairs):
            q, p = vectors[2*i:2*i+2]
            qt, pt = set(_terms(query)), set(_terms(passage))
            qn, pn = {t for t in qt if t.isdecimal()}, {t for t in pt if t.isdecimal()}
            lexical = len(qt & pt) / max(1, len(qt))
            numbers = len(qn & pn) / len(qn) if qn else 0.0
            # Counts/overlap are diagnostics, not linguistic proof or confidence.
            extra = [float(q @ p), lexical, numbers,
                     min(len(query), len(passage))/max(len(query), len(passage)),
                     min(len(qt), 128)/128, min(len(pt), 128)/128]
            output.append(np.concatenate((q, p, abs(q-p), q*p, extra)))
        return np.asarray(output, dtype=np.float64)


class NeuralRanker:
    """One-hidden-layer ReLU head, pairwise logistic relevance loss.

    Random initialization is deliberately unusable as a production reranker.
    Fit marks weights trained, NOT validated or release-ready. Encoder inference
    and its privacy/network behavior remain the caller's responsibility.
    """
    def __init__(self, features: PairFeatures, *, hidden_size: int = 64, seed: int = 0):
        np = _numpy()
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValidationError('Positive hidden_size required')
        self.features = features
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, math.sqrt(2/features.width), (features.width, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.w2 = rng.normal(0, math.sqrt(1/hidden_size), hidden_size)
        self.b2 = 0.0
        self.trained = False
        self.training_groups: frozenset[str] = frozenset()
        self.history: list[float] = []

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters."""
        return self.features.width*self.hidden_size + 2*self.hidden_size + 1

    def _forward(self, x: Any) -> tuple[Any, Any, Any]:
        np = _numpy()
        z = x @ self.w1 + self.b1
        h = np.maximum(z, 0)
        return h @ self.w2 + self.b2, z, h

    def _loss_grad(self, positive: Any, negative: Any, l2: float = 0.0) -> tuple[float, dict[str, Any]]:
        np = _numpy()
        sp, zp, hp = self._forward(positive)
        sn, zn, hn = self._forward(negative)
        diff = sp-sn
        loss = np.logaddexp(0, -diff).mean() + l2*(np.sum(self.w1**2)+np.sum(self.w2**2))/2
        # Stable d softplus(-diff) / d diff; batch-mean gradient.
        grad = -np.exp(-np.logaddexp(0, diff))/len(diff)
        gp = (grad[:, None]*self.w2) * (zp > 0)
        gn = (-grad[:, None]*self.w2) * (zn > 0)
        gradients = {'w1': positive.T@gp + negative.T@gn + l2*self.w1,
                     'b1': (gp+gn).sum(axis=0),
                     'w2': hp.T@grad-hn.T@grad+l2*self.w2,
                     'b2': 0.0}  # Pairwise loss cancels global output bias.
        return float(loss), gradients

    def fit(self, pairs: Sequence[ReviewedPair], *, held_out_groups: frozenset[str],
            epochs: int = 100, learning_rate: float = 0.01, l2: float = 0.001) -> list[float]:
        """Fit on reviewed pairs only; returns the loss history."""
        if not pairs or not isinstance(epochs, int) or epochs < 1:
            raise ValidationError('Nonempty reviewed pairs and positive epochs required')
        if (not math.isfinite(learning_rate) or learning_rate <= 0 or
                not math.isfinite(l2) or l2 < 0):
            raise ValidationError('Finite positive learning rate and nonnegative L2 required')
        if any(not isinstance(p, ReviewedPair) for p in pairs):
            raise ValidationError('Only ReviewedPair inputs accepted')
        groups = frozenset(p.event_group for p in pairs)
        if not held_out_groups or groups & held_out_groups:
            raise ValidationError('Explicit disjoint held-out event groups required')
        xp = self.features.transform([(p.query, p.positive) for p in pairs])
        xn = self.features.transform([(p.query, p.negative) for p in pairs])
        np = _numpy()
        # Compute on temporary copies so failure cannot publish partial weights.
        old = (self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2)
        history = []
        try:
            for _ in range(epochs):
                loss, gradients = self._loss_grad(xp, xn, l2)
                if not math.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients.values()):
                    raise ValidationError('Nonfinite training state')
                for name, gradient in gradients.items():
                    setattr(self, name, getattr(self, name)-learning_rate*gradient)
                if any(not np.isfinite(getattr(self, name)).all() for name in gradients):
                    raise ValidationError('Nonfinite updated weights')
                history.append(loss)
        except Exception:
            self.w1, self.b1, self.w2, self.b2 = old
            raise
        self.trained = True
        self.training_groups = groups
        self.history = history
        return list(history)

    def scores(self, query: str, evidence: tuple[Evidence, ...]) -> tuple[float, ...]:
        """Score each evidence item for the query; scores are not truth probabilities."""
        if not self.trained:
            raise ValidationError('Untrained ranker: no usable weights')
        if not evidence:
            return ()
        np = _numpy()
        values, _, _ = self._forward(self.features.transform([(query, e.excerpt) for e in evidence]))
        if not np.isfinite(values).all():
            raise ValidationError('Nonfinite ranker output')
        return tuple(float(x) for x in values)

    def __call__(self, query: str, evidence: tuple[Evidence, ...]) -> tuple[Evidence, ...]:
        values = self.scores(query, evidence)
        return tuple(evidence[i] for i in sorted(range(len(evidence)), key=lambda i: (-values[i], i)))

    def save(self, path: str | Path) -> None:
        """Write private weights only to a fresh directory; no source texts."""
        if not self.trained:
            raise ValidationError('Cannot export random/untrained weights')
        np = _numpy()
        path = Path(path)
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
        metadata = {'format': 'newsrag-neural-experimental-1',
                    'encoder_id': self.features.encoder_id,
                    'encoder_revision': self.features.encoder_revision,
                    'dimension': self.features.dimension, 'hidden_size': self.hidden_size,
                    'training_groups': sorted(self.training_groups), 'validated_for_release': False,
                    'scores_are_truth_probabilities': False}
        np.savez(path/'weights.npz', w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        (path/'metadata.json').write_text(json.dumps(metadata, indent=2))
        for filename in ('weights.npz', 'metadata.json'):
            (path/filename).chmod(0o600)


@dataclass(frozen=True)
class CalibrationCase:
    """One reviewed calibration case: score, whether relevant evidence exists, event group and review status."""
    score: float
    relevant_available: bool
    event_group: str
    review_status: str


class AbstentionGate:
    """Threshold calibrated for relevant evidence presence, NOT claim truth.

    Use disjoint calibration groups. Test groups must remain unseen. Caller must
    apply the gate outside the core's permutation-only reranker contract.
    """
    def __init__(self) -> None:
        self.threshold: float | None = None
        self.calibration_groups: frozenset[str] = frozenset()

    def calibrate(self, cases: Sequence[CalibrationCase], *, training_groups: frozenset[str],
                  held_out_groups: frozenset[str], max_false_accept_rate: float = 0.05) -> dict[str, Any]:
        """Pick the highest-recall threshold whose false-accept rate on reviewed, group-disjoint cases is within the bound."""
        if not cases or not held_out_groups or not math.isfinite(max_false_accept_rate) or not 0 <= max_false_accept_rate <= 1:
            raise ValidationError('Calibration cases, held-out groups and valid error bound required')
        for case in cases:
            if (case.review_status != 'independently_reviewed' or not case.event_group.strip() or
                    not math.isfinite(case.score) or not isinstance(case.relevant_available, bool)):
                raise ValidationError('Reviewed finite calibration cases required')
        groups = frozenset(c.event_group for c in cases)
        if groups & (training_groups | held_out_groups):
            raise ValidationError('Calibration groups must be disjoint from train/test')
        positive = [c for c in cases if c.relevant_available]
        negative = [c for c in cases if not c.relevant_available]
        if not positive or not negative:
            raise ValidationError('Positive and no-evidence calibration cases both required')
        candidates = sorted({c.score for c in cases}) + [math.inf]
        viable = []
        for t in candidates:
            false_rate = sum(c.score >= t for c in negative)/len(negative)
            recall = sum(c.score >= t for c in positive)/len(positive)
            if false_rate <= max_false_accept_rate:
                viable.append((recall, -false_rate, t))
        best = max(viable, key=lambda x: (x[0], x[1], x[2]))
        self.threshold = best[2]
        self.calibration_groups = groups
        return {'threshold': self.threshold, 'calibration_recall': best[0],
                'calibration_false_accept_rate': -best[1], 'held_out_performance': None}

    def accept(self, scores: Sequence[float]) -> bool:
        """True when the best score reaches the calibrated threshold (relevant evidence present, not claim truth)."""
        if self.threshold is None:
            raise ValidationError('Uncalibrated gate')
        if any(not math.isfinite(x) for x in scores):
            raise ValidationError('Finite scores required')
        return bool(scores) and max(scores) >= self.threshold


# ---------------------------------------------------------------------------
# TopicNet: experimental topic-understanding component (added 2026-10-25).
#
# Role (bounded, documented): TOPIC understanding of Arabic user input.
# Given an Arabic claim/question text, predict its topic category so the
# library can use it as an advisory *signal* for category-aware retrieval
# routing. It is NOT a verdict classifier and NOT an intent classifier, and
# it never overrides the deterministic evidence guards in ``assist()``;
# ``validated_for_release`` stays False until independent held-out evidence
# supports the claimed role. Current real-data evaluation (AraFacts,
# held-out-source split) shows it does NOT beat the deterministic tf-idf
# nearest-centroid baseline, so it remains exploratory.
#
# Architecture counting convention ("4 layers" = 4 learnable weight layers):
#   input (tf-idf, vocab V) -> h1 (W1,b1) tanh -> h2 (W2,b2) tanh
#                           -> h3 (W3,b3) tanh -> out (W4,b4) softmax
# Activations are not counted; only weight-bearing layers are. NumPy-only,
# CPU-trainable in under a minute at AraFacts scale (~4.6k rows).

import re as _re

_TOPIC_TOKEN_RE = _re.compile(r"[\u0621-\u064A\u0671-\u06D3a-zA-Z0-9]+")


def _tokenize(text: str):
    return _TOPIC_TOKEN_RE.findall((text or "").lower())


TOPIC_LAYER_COUNT = 4  # learnable weight layers W1..W4 (see note above)


class TopicNet:
    """One 4-layer (4 learnable weight layers) MLP over tf-idf features."""

    def __init__(self, vocab_size: int = 4000, hidden: tuple = (256, 128, 64), seed: int = 13):
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.seed = seed
        self.vocab: dict = {}
        self.idf = None
        self.weights: list = []
        self.biases: list = []
        self.classes_: list = []
        self.fitted = False
        self.validated_for_release = False  # signal-only; see module note

    def _build_vocab(self, texts):
        import math
        from collections import Counter
        df, tf_all = Counter(), Counter()
        for t in texts:
            toks = _tokenize(t)
            tf_all.update(toks)
            df.update(set(toks))
        keep = [w for w, _ in tf_all.most_common(self.vocab_size)]
        self.vocab = {w: i for i, w in enumerate(keep)}
        n = max(len(texts), 1)
        np_ = _numpy()
        self.idf = np_.array([math.log((1 + n) / (1 + df.get(w, 0))) + 1.0 for w in keep])

    def _vectorize(self, texts):
        import math
        from collections import Counter
        np_ = _numpy()
        V = len(self.vocab)
        X = np_.zeros((len(texts), V))
        for r, t in enumerate(texts):
            toks = _tokenize(t)
            if not toks:
                continue
            tf = Counter(toks)
            for w, c in tf.items():
                i = self.vocab.get(w)
                if i is not None:
                    X[r, i] = (1.0 + math.log(c)) * self.idf[i]
        return X

    def fit(self, texts, labels, epochs: int = 60, lr: float = 0.5, batch: int = 64):
        import math
        np_ = _numpy()
        self._build_vocab(texts)
        X = self._vectorize(texts)
        self.classes_ = sorted(set(labels))
        C = len(self.classes_)
        idx = {c: i for i, c in enumerate(self.classes_)}
        Y = np_.zeros((len(labels), C))
        for r, lab in enumerate(labels):
            Y[r, idx[lab]] = 1.0
        dims = [X.shape[1], *self.hidden, C]
        rng = np_.random.default_rng(self.seed)
        self.weights = [rng.normal(0, math.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                        for i in range(TOPIC_LAYER_COUNT)]
        self.biases = [np_.zeros(dims[i + 1]) for i in range(TOPIC_LAYER_COUNT)]
        n = X.shape[0]
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch):
                xb, yb = X[order[s:s + batch]], Y[order[s:s + batch]]
                acts = [xb]
                for i in range(TOPIC_LAYER_COUNT - 1):
                    acts.append(np_.tanh(acts[-1] @ self.weights[i] + self.biases[i]))
                z = acts[-1] @ self.weights[-1] + self.biases[-1]
                z -= z.max(axis=1, keepdims=True)
                p = np_.exp(z); p /= p.sum(axis=1, keepdims=True)
                delta = (p - yb) / len(xb)
                gw, gb = [], []
                a_prev = acts[-1]
                for i in reversed(range(TOPIC_LAYER_COUNT)):
                    gw.append(a_prev.T @ delta)
                    gb.append(delta.sum(axis=0))
                    if i > 0:
                        delta = (delta @ self.weights[i].T) * (1 - acts[i] ** 2)
                        a_prev = acts[i - 1]
                for i in range(TOPIC_LAYER_COUNT):
                    self.weights[i] -= lr * gw[TOPIC_LAYER_COUNT - 1 - i]
                    self.biases[i] -= lr * gb[TOPIC_LAYER_COUNT - 1 - i]
            lr *= 0.97
        self.fitted = True
        return {"classes": self.classes_, "vocab": len(self.vocab), "layers": TOPIC_LAYER_COUNT}

    def predict_proba(self, texts):
        np_ = _numpy()
        if not self.fitted:
            raise RuntimeError("TopicNet is not fitted; refusing to guess (fail-closed)")
        X = self._vectorize(texts)
        a = X
        for i in range(TOPIC_LAYER_COUNT - 1):
            a = np_.tanh(a @ self.weights[i] + self.biases[i])
        z = a @ self.weights[-1] + self.biases[-1]
        z -= z.max(axis=1, keepdims=True)
        p = np_.exp(z); p /= p.sum(axis=1, keepdims=True)
        return p

    def predict(self, texts):
        p = self.predict_proba(texts)
        return [self.classes_[i] for i in p.argmax(axis=1)]


def tfidf_centroid_baseline(train_texts, train_labels, test_texts, seed: int = 13):
    """Deterministic non-neural baseline: tf-idf nearest-centroid cosine.
    Shares tokenization/vocab protocol with TopicNet for a fair comparison."""
    np_ = _numpy()
    net = TopicNet(seed=seed)
    net._build_vocab(train_texts)
    Xtr = net._vectorize(train_texts)
    Xte = net._vectorize(test_texts)
    classes = sorted(set(train_labels))
    cents = []
    for c in classes:
        rows = [i for i, l in enumerate(train_labels) if l == c]
        cents.append(Xtr[rows].mean(axis=0))
    C = np_.array(cents)
    na = np_.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9
    nb = np_.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    sims = (Xte / na) @ (C / nb).T
    return [classes[i] for i in sims.argmax(axis=1)]
