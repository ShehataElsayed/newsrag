"""Experimental weak-label rank fusion and honest evaluation helpers.

Nothing here verifies claims. The fusion ranker learns weights over caller-given
score features from weak or professional labels and keeps that origin. Dates,
verdict maps and splits are structural checks, not proof of event independence.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .core import ValidationError
from .neural import _numpy

_SOURCES = frozenset({'independently_reviewed', 'weak_title_body', 'external_professional'})


def ranks_from_scores(scores: Sequence[Sequence[float]], targets: Sequence[int]) -> list[int]:
    """1-based rank of each target; ties count against the target (pessimistic)."""
    if len(scores) != len(targets) or not scores:
        raise ValidationError('درجات وأهداف بعدد متساو وغير فارغ مطلوبة')
    out = []
    for row, target in zip(scores, targets, strict=True):
        if not 0 <= target < len(row) or any(not math.isfinite(x) for x in row):
            raise ValidationError('هدف خارج النطاق أو درجة غير محدودة')
        out.append(1 + sum(1 for x in row if x >= row[target]) - 1)
    return out


def ranking_metrics(ranks: Sequence[int]) -> dict[str, float]:
    """Top-1, top-5 and MRR from 1-based ranks."""
    if not ranks or any(type(r) is not int or r < 1 for r in ranks):
        raise ValidationError('رتب صحيحة موجبة مطلوبة')
    n = len(ranks)
    return {'top1': sum(r == 1 for r in ranks) / n, 'top5': sum(r <= 5 for r in ranks) / n,
            'mrr': sum(1 / r for r in ranks) / n}


@dataclass(frozen=True)
class PairedDifference:
    """A paired mean difference with a 95% interval over cases."""
    mean: float
    low: float
    high: float
    cases: int

    @property
    def excludes_zero(self) -> bool:
        """True when the 95% interval does not contain zero."""
        return self.low > 0 or self.high < 0


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, resamples: int = 2000,
                     seed: int = 7) -> PairedDifference:
    """Mean paired difference a-b with a 95% percentile interval over cases."""
    if len(a) != len(b) or len(a) < 2 or resamples < 100:
        raise ValidationError('قياسات مزدوجة متساوية (اثنان فأكثر) وإعادات كافية مطلوبة')
    np = _numpy()
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    return PairedDifference(float(d.mean()), float(np.percentile(means, 2.5)),
                            float(np.percentile(means, 97.5)), len(d))


def temporal_split(items: Sequence[Any], date_of: Callable[[Any], str], cutoff: str
                   ) -> tuple[list[Any], list[Any]]:
    """Before cutoff -> train, on/after -> test. ISO dates only; empty sides rejected."""
    try:
        boundary = date.fromisoformat(cutoff)
        stamps = [date.fromisoformat(date_of(i)[:10]) for i in items]
    except ValueError as error:
        raise ValidationError('تواريخ ISO صالحة مطلوبة') from error
    train = [i for i, s in zip(items, stamps, strict=True) if s < boundary]
    test = [i for i, s in zip(items, stamps, strict=True) if s >= boundary]
    if not train or not test:
        raise ValidationError('كل جانب من التقسيم الزمني يجب ألا يكون فارغا')
    return train, test


def _squash(text: str) -> str:
    return ''.join(ch for ch in text if ch.isalnum())


def duplicate_rate(train_texts: Sequence[str], test_texts: Sequence[str]) -> float:
    """Share of test texts equal to a train text after dropping non-alphanumerics.

    Zero does not prove no near-duplicates or same-event leakage."""
    if not train_texts or not test_texts:
        raise ValidationError('نصوص تدريب واختبار غير فارغة مطلوبة')
    seen = {_squash(t) for t in train_texts}
    return sum(_squash(t) in seen for t in test_texts) / len(test_texts)


@dataclass(frozen=True)
class VerdictMapping:
    """Per-platform label map. Equivalence between platforms is NOT asserted."""
    platform: str
    labels: Mapping[str, str]
    caveat: str

    def __post_init__(self) -> None:
        if (not self.platform.strip() or not self.labels or not self.caveat.strip()
                or any(not k.strip() or not v.strip() for k, v in self.labels.items())):
            raise ValidationError('منصة وتعيين تصنيفات وتحذير تكافؤ صريح مطلوبة')

    def normalize(self, label: str) -> str:
        """Map a platform label to its normalized verdict; unknown labels raise ValidationError."""
        try:
            return self.labels[label]
        except KeyError as error:
            raise ValidationError(f'تصنيف غير معروف عند {self.platform}: {label}') from error


class LinearFusionRanker:
    """Listwise softmax ranker over standardized score features (NumPy, no MLP).

    fit() expects, per query, candidate rows with the true candidate first.
    Weights start at 1 so training begins from the plain feature sum."""

    def __init__(self, feature_names: Sequence[str], *, feature_revision: str):
        if (not feature_names or len(set(feature_names)) != len(feature_names)
                or not feature_revision.strip()):
            raise ValidationError('أسماء خصائص فريدة ونسخة خصائص مطلوبة')
        self.feature_names = tuple(feature_names)
        self.feature_revision = feature_revision
        self.weights: Any = None
        self.mean: Any = None
        self.std: Any = None
        self.label_sources: frozenset[str] = frozenset()
        self.history: list[float] = []

    @property
    def trained(self) -> bool:
        """True once the ranker has been fitted."""
        return self.weights is not None

    def fit(self, queries: Sequence[Sequence[Sequence[float]]], *, label_source: str,
            epochs: int = 200, learning_rate: float = 0.05, l2: float = 1e-3) -> list[float]:
        """Fit weights on queries of candidate feature rows with the given label origin; returns the loss history."""
        if label_source not in _SOURCES:
            raise ValidationError('أصل وسم صريح مطلوب')
        np = _numpy()
        width = len(self.feature_names)
        if not queries or any(len(q) < 2 or any(len(r) != width for r in q) for q in queries):
            raise ValidationError('كل استعلام يحتاج مرشحين على الأقل بعرض الخصائص الصحيح')
        k = {len(q) for q in queries}
        if len(k) != 1:
            raise ValidationError('عدد المرشحين لكل استعلام يجب أن يتساوى')
        x = np.asarray(queries, dtype=float)
        if not np.isfinite(x).all() or epochs < 1 or learning_rate <= 0:
            raise ValidationError('قيم محدودة وإعدادات تدريب صحيحة مطلوبة')
        flat = x.reshape(-1, width)
        self.mean, self.std = flat.mean(axis=0), flat.std(axis=0) + 1e-9
        z = (x - self.mean) / self.std
        w = np.ones(width)
        hist = []
        for _ in range(epochs):
            logits = z @ w
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            hist.append(float(-np.log(p[:, 0] + 1e-12).mean()) + l2 * float((w ** 2).sum()))
            onehot = np.zeros_like(p)
            onehot[:, 0] = 1
            grad = ((p - onehot)[:, :, None] * z).sum(axis=1).mean(axis=0) + 2 * l2 * w
            w = w - learning_rate * grad
        self.weights = w
        self.label_sources = frozenset({label_source})
        self.history = hist
        return hist

    def scores(self, features: Sequence[Sequence[float]]) -> list[float]:
        """Score the candidate feature rows (higher ranks first)."""
        if not self.trained:
            raise ValidationError('المرتب غير مدرب؛ لا تنبؤ')
        np = _numpy()
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValidationError('مصفوفة خصائص محدودة بالعرض الصحيح مطلوبة')
        return [float(v) for v in ((x - self.mean) / self.std) @ self.weights]

    def save(self, directory: str | Path) -> None:
        """Write the fitted weights, label origin and file hash to disk."""
        if not self.trained:
            raise ValidationError('لا حفظ لمرتب غير مدرب')
        path = Path(directory)
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ValidationError('مجلد الحفظ يجب أن يكون جديدا') from error
        np = _numpy()
        np.savez(path / 'weights.npz', weights=self.weights, mean=self.mean, std=self.std)
        digest = hashlib.sha256((path / 'weights.npz').read_bytes()).hexdigest()
        meta = {'kind': 'linear_fusion_ranker', 'feature_names': list(self.feature_names),
                'feature_revision': self.feature_revision, 'label_sources': sorted(self.label_sources),
                'trained': True, 'validated_for_release': False, 'truth_probability': None,
                'weights_sha256': digest}
        (path / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding='utf-8')

    @classmethod
    def load(cls, directory: str | Path, *, feature_names: Sequence[str],
             feature_revision: str) -> LinearFusionRanker:
        """Read weights back and verify the stored hash before use."""
        path = Path(directory)
        weights_file, meta_file = path / 'weights.npz', path / 'metadata.json'
        if not weights_file.is_file() or not meta_file.is_file() or weights_file.stat().st_size > 1_000_000:
            raise ValidationError('ملفات أوزان وبيانات وصفية صالحة مطلوبة')
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
        if (meta.get('kind') != 'linear_fusion_ranker' or meta.get('trained') is not True
                or meta.get('feature_names') != list(feature_names)
                or meta.get('feature_revision') != feature_revision
                or not re.fullmatch(r'[0-9a-f]{64}', str(meta.get('weights_sha256', '')))
                or hashlib.sha256(weights_file.read_bytes()).hexdigest() != meta['weights_sha256']):
            raise ValidationError('بيانات وصفية أو بصمة أو خصائص غير مطابقة')
        np = _numpy()
        ranker = cls(feature_names, feature_revision=feature_revision)
        with np.load(weights_file, allow_pickle=False) as arrays:
            w, m, s = arrays['weights'], arrays['mean'], arrays['std']
        n = len(feature_names)
        if w.shape != (n,) or m.shape != (n,) or s.shape != (n,) or not all(
                np.isfinite(a).all() for a in (w, m, s)):
            raise ValidationError('أشكال أوزان غير صحيحة')
        ranker.weights, ranker.mean, ranker.std = w, m, s
        ranker.label_sources = frozenset(meta['label_sources'])
        return ranker
