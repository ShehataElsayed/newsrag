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
