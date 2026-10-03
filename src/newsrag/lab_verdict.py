"""Experimental triage verdict component with calibration and abstention.

It predicts the label vocabulary of ONE labeling source from claim text. It does
not check facts, never acts alone on weak or external labels, and always hands
the final call to an editor. Autonomous use stays off unless thresholds were
measured on independently reviewed data (not available in this project).
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from .core import ValidationError
from .lab_stats import wilson_interval
from .neural import _numpy

MIN_REVIEWED_FOR_AUTONOMY = 500
_DIACRITICS = re.compile('[\u064b-\u065f\u0670\u0640]')
_SPACE = re.compile(r'\s+')


def normalize_claim(text: str) -> str:
    """Light Arabic normalization: diacritics, tatweel, alef/ya/ta-marbuta variants."""
    text = unicodedata.normalize('NFKC', text)
    text = _DIACRITICS.sub('', text)
    for src, dst in (('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'), ('ى', 'ي'), ('ة', 'ه')):
        text = text.replace(src, dst)
    return _SPACE.sub(' ', text).strip().lower()


def _features(text: str, ngram: tuple[int, int]) -> list[str]:
    norm = normalize_claim(text)
    words = norm.split()
    grams = ['w:' + w for w in words] + ['b:' + a + '_' + b for a, b in pairwise(words)]
    padded = f' {norm} '
    for n in range(ngram[0], ngram[1] + 1):
        grams.extend('c:' + padded[i:i + n] for i in range(len(padded) - n + 1))
    return grams


def _bucket(token: str, dim: int) -> int:
    value = 2166136261
    for byte in token.encode('utf-8'):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value % dim


class HashedTfidf:
    """Hashed word, bigram and character n-gram TF-IDF (sublinear, L2 normalized)."""

    def __init__(self, dim: int = 4096, ngram: tuple[int, int] = (3, 5)):
        if dim < 64 or ngram[0] < 1 or ngram[1] < ngram[0]:
            raise ValidationError('بُعد أو مدى n-gram غير صالح')
        self.dim, self.ngram = dim, ngram
        self.idf: Any = None

    def _counts(self, texts: Sequence[str]) -> list[Counter[int]]:
        return [Counter(_bucket(g, self.dim) for g in _features(t, self.ngram)) for t in texts]

    def fit(self, texts: Sequence[str]) -> HashedTfidf:
        """Learn the IDF weights from the texts and return self."""
        if not texts:
            raise ValidationError('نصوص تدريب مطلوبة')
        np = _numpy()
        df = np.zeros(self.dim)
        for c in self._counts(texts):
            for k in c:
                df[k] += 1
        self.idf = np.log((1 + len(texts)) / (1 + df)) + 1.0
        return self

    def transform(self, texts: Sequence[str]) -> Any:
        """Map texts to L2-normalised hashed TF-IDF vectors."""
        if self.idf is None:
            raise ValidationError('المحوّل غير مدرَّب')
        np = _numpy()
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, c in enumerate(self._counts(texts)):
            for k, v in c.items():
                out[i, k] = (1 + math.log(v)) * self.idf[k]
            norm = float(np.linalg.norm(out[i]))
            if norm:
                out[i] /= norm
        return out


def softmax(logits: Any) -> Any:
    """Row-wise numerically stable softmax."""
    np = _numpy()
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxClassifier:
    """Multinomial logistic regression (full-batch Adam, L2, optional class weights)."""

    def __init__(self, classes: Sequence[str], *, l2: float = 1e-4, epochs: int = 300,
                 lr: float = 0.05, balanced: bool = True, seed: int = 7):
        if len(set(classes)) != len(classes) or len(classes) < 2:
            raise ValidationError('صنفان مختلفان على الأقل مطلوبان')
        self.classes = tuple(classes)
        self.l2, self.epochs, self.lr, self.balanced, self.seed = l2, epochs, lr, balanced, seed
        self.w: Any = None
        self.b: Any = None
        self.loss_history: list[float] = []

    def fit(self, x: Any, labels: Sequence[str]) -> SoftmaxClassifier:
        """Fit by full-batch Adam on feature rows and labels and return self."""
        np = _numpy()
        index = {c: i for i, c in enumerate(self.classes)}
        try:
            y = np.array([index[v] for v in labels])
        except KeyError as error:
            raise ValidationError(f'صنف غير معروف: {error}') from error
        if len(y) != len(x) or not len(y):
            raise ValidationError('ميزات وأصناف بعدد متساو وغير فارغ مطلوبة')
        k = len(self.classes)
        counts = np.bincount(y, minlength=k).astype(float)
        sw = (len(y) / (k * np.maximum(counts, 1)))[y] if self.balanced else np.ones(len(y))
        sw = sw / sw.mean()
        rng = np.random.default_rng(self.seed)
        w = rng.normal(0, 0.01, (x.shape[1], k)).astype(np.float32); b = np.zeros(k, dtype=np.float32)
        m = [np.zeros_like(w), np.zeros_like(b)]; v = [np.zeros_like(w), np.zeros_like(b)]
        onehot = np.eye(k, dtype=np.float32)[y]
        swf = sw.astype(np.float32)
        for t in range(1, self.epochs + 1):
            p = softmax(x @ w + b).astype(np.float32)
            loss = float(-(sw * np.log(p[np.arange(len(y)), y] + 1e-12)).mean() + 0.5 * self.l2 * (w ** 2).sum())
            self.loss_history.append(loss)
            g = ((p - onehot) * swf[:, None] / len(y)).astype(np.float32)
            grads = [x.T @ g + np.float32(self.l2) * w, g.sum(axis=0)]
            for i, (param, grad) in enumerate(zip((w, b), grads, strict=True)):
                m[i] = 0.9 * m[i] + 0.1 * grad
                v[i] = 0.999 * v[i] + 0.001 * grad ** 2
                param -= self.lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        self.w, self.b = w, b
        return self

    def logits(self, x: Any) -> Any:
        """Return raw class scores for the given inputs."""
        if self.w is None:
            raise ValidationError('المصنف غير مدرَّب')
        return x @ self.w + self.b


def fit_temperature(logits: Any, labels_index: Sequence[int]) -> float:
    """Single temperature minimizing NLL on a calibration split (grid search)."""
    np = _numpy()
    y = np.asarray(labels_index)
    if len(y) != len(logits) or not len(y):
        raise ValidationError('درجات وأصناف بعدد متساو وغير فارغ مطلوبة')
    best, best_t = None, 1.0
    for t in np.linspace(0.3, 5.0, 95):
        p = softmax(logits / t)
        nll = float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())
        if best is None or nll < best:
            best, best_t = nll, float(t)
    return best_t


def risk_coverage(confidence: Sequence[float], correct: Sequence[bool]) -> list[tuple[float, float, float]]:
    """(threshold, coverage, selective accuracy) from most to least confident."""
    if len(confidence) != len(correct) or not confidence:
        raise ValidationError('ثقة وصحة بعدد متساو وغير فارغ مطلوبة')
    order = sorted(range(len(confidence)), key=lambda i: -confidence[i])
    out, hits = [], 0
    for rank, i in enumerate(order, start=1):
        hits += bool(correct[i])
        if rank == len(order) or confidence[order[rank]] != confidence[i]:
            out.append((confidence[i], rank / len(order), hits / rank))
    return out


def choose_threshold(confidence: Sequence[float], correct: Sequence[bool], *,
                     target_accuracy: float, min_cases: int = 30) -> float:
    """Lowest threshold whose Wilson LOWER bound on selective accuracy meets the target.

    Returns +inf (always abstain) when no threshold qualifies with enough cases."""
    if not 0 < target_accuracy < 1:
        raise ValidationError('دقة مستهدفة بين 0 و1 مطلوبة')
    order = sorted(range(len(confidence)), key=lambda i: -confidence[i])
    best = math.inf
    hits = 0
    for rank, i in enumerate(order, start=1):
        hits += bool(correct[i])
        tie_end = rank == len(order) or confidence[order[rank]] != confidence[i]
        if tie_end and rank >= min_cases and wilson_interval(hits, rank).low >= target_accuracy:
            best = confidence[i]
    return best


@dataclass(frozen=True)
class VerdictDecision:
    """Always a suggestion. `action` is never an autonomous publication step."""
    suggestion: str | None
    confidence: float
    abstained: bool
    probabilities: dict[str, float]
    label_source: str
    action: str = 'editor_review_required'
    validated_for_release: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


class AbstainingVerdict:
    """Classifier + temperature + abstention threshold, honest about its label source."""

    def __init__(self, classes: Sequence[str], *, label_source: str, platform: str,
                 equivalence_caveat: str, dim: int = 4096, **classifier_options: Any):
        if label_source not in {'external_professional', 'independently_reviewed', 'weak_title_body'}:
            raise ValidationError('أصل وسم غير معروف')
        if not platform.strip() or not equivalence_caveat.strip():
            raise ValidationError('منصة وتحذير تكافؤ صريح مطلوبان')
        self.label_source, self.platform, self.caveat = label_source, platform, equivalence_caveat
        self.vectorizer = HashedTfidf(dim)
        self.model = SoftmaxClassifier(classes, **classifier_options)
        self.temperature = 1.0
        self.threshold = math.inf
        self.reviewed_cases = 0
        self.target_accuracy: float | None = None

    @property
    def classes(self) -> tuple[str, ...]:
        """The class labels in column order."""
        return self.model.classes

    @property
    def autonomous_enabled(self) -> bool:
        """True only with independent labels, at least 500 reviewed cases and a finite threshold."""
        return (self.label_source == 'independently_reviewed'
                and self.reviewed_cases >= MIN_REVIEWED_FOR_AUTONOMY
                and math.isfinite(self.threshold))

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> AbstainingVerdict:
        """Fit the vectorizer and classifier on texts and labels and return self."""
        self.model.fit(self.vectorizer.fit(texts).transform(texts), labels)
        return self

    def logits(self, texts: Sequence[str]) -> Any:
        """Return raw class scores for the given inputs."""
        return self.model.logits(self.vectorizer.transform(texts))

    def probabilities(self, texts: Sequence[str]) -> Any:
        """Return calibrated class probabilities."""
        return softmax(self.logits(texts) / self.temperature)

    def calibrate(self, texts: Sequence[str], labels: Sequence[str], *,
                  target_accuracy: float = 0.9, min_cases: int = 30) -> AbstainingVerdict:
        """Fit temperature, then the abstain threshold on a split NOT used for training."""
        np = _numpy()
        index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([index[v] for v in labels])
        logits = self.logits(texts)
        self.temperature = fit_temperature(logits, y)
        p = softmax(logits / self.temperature)
        conf = p.max(axis=1).tolist()
        ok = (p.argmax(axis=1) == y).tolist()
        self.threshold = choose_threshold(conf, ok, target_accuracy=target_accuracy, min_cases=min_cases)
        self.target_accuracy = target_accuracy
        if self.label_source == 'independently_reviewed':
            self.reviewed_cases = len(labels)
        return self

    def decide(self, text: str) -> VerdictDecision:
        """Return a VerdictDecision: a suggestion or an abstention, always for editor review."""
        p = self.probabilities([text])[0]
        top = int(p.argmax())
        conf = float(p[top])
        abstain = conf < self.threshold
        notes = [f'تنبؤ بمفردات تصنيف {self.platform} فقط، لا تحقق من الوقائع', self.caveat]
        if not self.autonomous_enabled:
            notes.append('القرار الآلي معطل: لا عتبات مقاسة على بيانات مراجَعة مستقلة')
        return VerdictDecision(
            suggestion=None if abstain else self.classes[top], confidence=conf, abstained=abstain,
            probabilities={c: float(v) for c, v in zip(self.classes, p, strict=True)},
            label_source=self.label_source, notes=tuple(notes))
