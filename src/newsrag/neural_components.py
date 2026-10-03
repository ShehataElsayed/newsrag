"""Experimental trainable heads for stance, token spans and source dependence.

Feature arrays must come from a caller-selected frozen encoder or observed graph
features. No pretrained weights, automated fact verdicts, identity guarantees or
release-quality claims are provided. Weak labels remain explicitly weak.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import ValidationError
from .neural import _numpy

_LABEL_SOURCES = frozenset({'independently_reviewed', 'weak_title_body',
                            'weak_surface_rule', 'synthetic_test_fixture', 'external_professional'})


@dataclass(frozen=True)
class LabeledFeature:
    """A feature vector with its label and label origin."""
    features: tuple[float, ...]
    label: int
    event_group: str
    label_source: str
    source_reference: str | None = None

    def __post_init__(self) -> None:
        if (not self.features or not all(math.isfinite(x) for x in self.features)
                or type(self.label) is not int or not self.event_group.strip()
                or self.label_source not in _LABEL_SOURCES):
            raise ValidationError('خصائص محدودة ووسم ومجموعة وأصل وسم صريح مطلوبة')
        if self.label_source == 'external_professional' and (
            not isinstance(self.source_reference, str) or not self.source_reference.strip()
        ):
            raise ValidationError('الوسم المهني الخارجي يحتاج مرجع المصدر الأصلي')


class DenseTaskHead:
    """ReLU MLP with multiclass cross-entropy and a manual NumPy gradient.

    Train and held-out groups cannot overlap. This is a structural gate; a caller
    inventing group IDs does not make the split event-disjoint. Calibration and
    semantic performance remain separate acceptance steps. Outputs are task
    class scores, not truth probabilities. Untrained heads cannot predict.
    """
    labels: tuple[str, ...] = ()
    task: str = 'unspecified'

    def __init__(self, *, input_width: int, hidden_size: int = 64, seed: int = 0,
                 feature_id: str, feature_revision: str):
        if (type(input_width) is not int or input_width < 1 or type(hidden_size) is not int
                or hidden_size < 1 or not feature_id.strip() or not feature_revision.strip()
                or len(self.labels) < 2):
            raise ValidationError('أبعاد ونسخة خصائص وفئات مهمة صريحة مطلوبة')
        np = _numpy()
        self.input_width, self.hidden_size = input_width, hidden_size
        self.feature_id, self.feature_revision = feature_id, feature_revision
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, math.sqrt(2/input_width), (input_width, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.w2 = rng.normal(0, math.sqrt(1/hidden_size), (hidden_size, len(self.labels)))
        self.b2 = np.zeros(len(self.labels))
        self.trained = False
        self.training_groups: frozenset[str] = frozenset()
        self.label_sources: frozenset[str] = frozenset()
        self.history: list[float] = []
        self.label_references: frozenset[str] = frozenset()

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters."""
        return self.input_width*self.hidden_size + self.hidden_size + (
            self.hidden_size*len(self.labels)) + len(self.labels)

    def _array(self, rows: Any) -> Any:
        np = _numpy()
        array = np.asarray(rows, dtype=np.float64)
        if (array.ndim != 2 or array.shape[1] != self.input_width
                or not np.isfinite(array).all()):
            raise ValidationError('مصفوفة خصائص محدودة ومتسقة الأبعاد مطلوبة')
        return array

    def _forward(self, x: Any) -> tuple[Any, Any, Any]:
        np = _numpy()
        z = x @ self.w1 + self.b1
        h = np.maximum(z, 0)
        logits = h @ self.w2 + self.b2
        shifted = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(shifted)
        p /= p.sum(axis=1, keepdims=True)
        return p, z, h

    def _loss_grad(self, x: Any, labels: Any, l2: float = 0.0) -> tuple[float, dict[str, Any]]:
        np = _numpy()
        p, z, h = self._forward(x)
        # Stable log-sum-exp from logits, without clipped probability gradients.
        logits = h @ self.w2 + self.b2
        maximum = logits.max(axis=1)
        logden = maximum + np.log(np.exp(logits-maximum[:, None]).sum(axis=1))
        loss = (logden-logits[np.arange(len(x)), labels]).mean()
        loss += l2*(np.sum(self.w1**2)+np.sum(self.w2**2))/2
        dlogits = p.copy()
        dlogits[np.arange(len(x)), labels] -= 1
        dlogits /= len(x)
        hidden = (dlogits @ self.w2.T) * (z > 0)
        return float(loss), {'w1': x.T @ hidden + l2*self.w1,
                             'b1': hidden.sum(axis=0),
                             'w2': h.T @ dlogits + l2*self.w2,
                             'b2': dlogits.sum(axis=0)}

    def fit(self, cases: Sequence[LabeledFeature], *, held_out_groups: frozenset[str],
            epochs: int = 100, learning_rate: float = .01, l2: float = .001) -> list[float]:
        """Fit on labelled features with explicit held-out groups; returns the loss history."""
        if (not cases or type(epochs) is not int or epochs < 1 or not held_out_groups
                or any(not isinstance(c, LabeledFeature) for c in cases)
                or not math.isfinite(learning_rate) or learning_rate <= 0
                or not math.isfinite(l2) or l2 < 0):
            raise ValidationError('عينات ومدة ومعاملات تدريب ومجموعات اختبار صريحة مطلوبة')
        groups = frozenset(c.event_group for c in cases)
        if groups & held_out_groups:
            raise ValidationError('تداخل مجموعات التدريب والاختبار غير مسموح')
        if any(not 0 <= c.label < len(self.labels) for c in cases):
            raise ValidationError('وسم خارج فئات المهمة')
        x = self._array([c.features for c in cases])
        np = _numpy()
        y = np.asarray([c.label for c in cases], dtype=np.int64)
        old = {name: getattr(self, name).copy() for name in ('w1', 'b1', 'w2', 'b2')}
        history = []
        try:
            for _ in range(epochs):
                loss, gradients = self._loss_grad(x, y, l2)
                if not math.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients.values()):
                    raise ValidationError('حالة تدريب غير محدودة')
                for name, gradient in gradients.items():
                    setattr(self, name, getattr(self, name)-learning_rate*gradient)
                if any(not np.isfinite(getattr(self, name)).all() for name in old):
                    raise ValidationError('أوزان غير محدودة')
                history.append(loss)
        except Exception:
            for name, value in old.items():
                setattr(self, name, value)
            raise
        self.trained, self.training_groups = True, groups
        self.label_sources = frozenset(c.label_source for c in cases)
        self.history = history
        self.label_references = frozenset(c.source_reference for c in cases
                                          if c.source_reference is not None)
        return list(history)

    def class_scores(self, rows: Any) -> Any:
        """Return per-class scores for one feature vector."""
        if not self.trained:
            raise ValidationError('الرأس غير مدرب؛ لا ينتج تنبؤات')
        np = _numpy()
        x = self._array(rows)
        if not len(x):
            return np.empty((0, len(self.labels)))
        p, _, _ = self._forward(x)
        if not np.isfinite(p).all():
            raise ValidationError('مخرجات غير محدودة')
        return p

    def status(self) -> dict[str, Any]:
        """Return the honest status of this component (trained or not, label origin, limits)."""
        return {'task': self.task, 'labels': self.labels, 'trained': self.trained,
                'parameter_count': self.parameter_count,
                'feature_id': self.feature_id, 'feature_revision': self.feature_revision,
                'label_sources': sorted(self.label_sources),
                'label_references': sorted(self.label_references),
                'human_reviewed_only': bool(self.label_sources) and self.label_sources == {
                    'independently_reviewed'}, 'validated_for_release': False,
                'truth_probability': None}


    def save(self, directory: str | Path) -> None:
        """Private fresh directory, weights only, never texts or random weights.

        Hash detects accidental corruption, not authorship. Metadata preserves
        weak/external origin. A saved model is not automatically accepted.
        """
        if not self.trained:
            raise ValidationError('لا تحفظ أوزانا عشوائية غير مدربة')
        np = _numpy()
        path = Path(directory)
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
        weights = path/'weights.npz'
        np.savez(weights, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        weights.chmod(0o600)
        metadata = {**self.status(), 'format': 'newsrag-task-head-experimental-1',
                    'input_width': self.input_width, 'hidden_size': self.hidden_size,
                    'training_groups': sorted(self.training_groups),
                    'history': self.history,
                    'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}
        manifest = path/'metadata.json'
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest.chmod(0o600)

    @classmethod
    def load(cls, directory: str | Path, *, feature_id: str,
             feature_revision: str) -> DenseTaskHead:
        """Read bounded non-pickle arrays with exact task/feature matching.

        Feature IDs are caller assertions. Hash is not a signature; never load
        untrusted model artifacts as an authorization or a quality guarantee.
        """
        import zipfile

        np = _numpy()
        path = Path(directory)
        manifest, weights = path/'metadata.json', path/'weights.npz'
        if manifest.stat().st_size > 2_000_000 or weights.stat().st_size > 64_000_000:
            raise ValidationError('ملف أوزان يتجاوز حدود القراءة')
        try:
            metadata = json.loads(manifest.read_text(encoding='utf-8'))
            if (metadata['format'] != 'newsrag-task-head-experimental-1'
                    or metadata['task'] != cls.task or tuple(metadata['labels']) != cls.labels
                    or metadata['feature_id'] != feature_id
                    or metadata['feature_revision'] != feature_revision
                    or metadata['trained'] is not True
                    or metadata['validated_for_release'] is not False
                    or metadata['truth_probability'] is not None
                    or metadata['weights_sha256'] != hashlib.sha256(weights.read_bytes()).hexdigest()):
                raise ValidationError('المهمة أو النسخة أو البصمة أو حالة الأوزان غير مطابقة')
            width, hidden = metadata['input_width'], metadata['hidden_size']
            if (type(width) is not int or type(hidden) is not int or width < 1 or hidden < 1
                    or width*hidden+hidden*len(cls.labels) > 4_000_000):
                raise ValidationError('أبعاد أوزان غير صالحة')
            groups, sources, references = (metadata['training_groups'], metadata['label_sources'],
                                          metadata['label_references'])
            if (not groups or not sources
                    or any(not isinstance(x, str) or not x.strip() for x in groups+sources+references)
                    or not set(sources) <= _LABEL_SOURCES
                    or ('external_professional' in sources and not references)):
                raise ValidationError('بيانات أصل التدريب غير صالحة')
            with zipfile.ZipFile(weights) as archive:
                if (set(archive.namelist()) != {'w1.npy', 'b1.npy', 'w2.npy', 'b2.npy'}
                        or sum(x.file_size for x in archive.infolist()) > 64_000_000):
                    raise ValidationError('بنية ملف أوزان غير صالحة')
            head = cls(input_width=width, hidden_size=hidden,
                       feature_id=feature_id, feature_revision=feature_revision)
            shapes = {'w1': (width, hidden), 'b1': (hidden,),
                      'w2': (hidden, len(cls.labels)), 'b2': (len(cls.labels),)}
            with np.load(weights, allow_pickle=False) as arrays:
                for name, shape in shapes.items():
                    value = arrays[name]
                    if (value.shape != shape or value.dtype.kind != 'f'
                            or not np.isfinite(value).all()):
                        raise ValidationError('مصفوفة أوزان غير صالحة')
                    setattr(head, name, value.copy())
            history = metadata['history']
            if not history or any(not isinstance(x, (int, float)) or not math.isfinite(x)
                                  for x in history):
                raise ValidationError('سجل تدريب غير صالح')
            head.trained = True
            head.training_groups, head.label_sources = frozenset(groups), frozenset(sources)
            head.label_references, head.history = frozenset(references), list(history)
            return head
        except (KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
            raise ValidationError('تعذر التحقق من ملف الرأس المدرب') from exc


class StanceHead(DenseTaskHead):
    """Claim-passage task. Scores never become automatic claim truth verdicts."""
    task = 'claim_passage_stance'
    labels = ('supports', 'refutes', 'insufficient')


class SpanTokenHead(DenseTaskHead):
    """Per-token span membership. Offset-safe decoding is a separate control.

    The head alone does not enforce a contiguous span or provide token offsets.
    Token features/offset maps must be supplied and checked by the caller.
    """
    task = 'evidence_token_membership'
    labels = ('outside', 'inside')


class SourceDependenceHead(DenseTaskHead):
    """Pairwise source-dependence head, not an independence guarantee.

    Graph traversal and event evidence are not implemented by this MLP. Features
    must encode an explicitly documented pair; raw cosine is not a gold label.
    """
    task = 'source_pair_dependence'
    labels = ('unknown_or_independent', 'dependent')


class RelevanceAvailabilityHead(DenseTaskHead):
    """Pair relevance head; weak title/body labels are not a calibrated gate.

    Independent calibration is required before this can inform abstention.
    Predicting an associated document does not establish support for a claim.
    """
    task = 'pair_relevance_availability'
    labels = ('not_associated', 'associated')


class ClaimTokenHead(DenseTaskHead):
    """Token membership in a checkable-claim candidate, not a truth verdict.

    Caller supplies token features, raw offsets and reviewed boundaries. Claim
    segmentation/decoding is separate. A positive token is only a candidate.
    """
    task = 'checkable_claim_token_membership'
    labels = ('outside', 'claim_candidate')


class ClaimStructureTokenHead(DenseTaskHead):
    """Token role proposals, with no relation extraction or entity resolution.

    Single labels cannot represent nested/overlapping roles. No implicit choice
    of a claim's subject, time, unit or denominator is authorized by scores.
    """
    task = 'claim_structure_token_role'
    labels = ('other', 'entity', 'predicate', 'time', 'quantity', 'unit')


class QuantityAlignmentHead(DenseTaskHead):
    """Pair quantity compatibility conditioned on caller's entity/unit context.

    Does not parse, convert or normalize units. Features and gold labels must
    distinguish counts, denominators, currencies, dates and measurements.
    """
    task = 'quantity_pair_alignment'
    labels = ('compatible', 'incompatible', 'insufficient_context')


class TemporalRelationHead(DenseTaskHead):
    """Pair temporal compatibility; publishing date alone is insufficient.

    Does not resolve relative dates, validity intervals or updated versions.
    Those must be supplied with provenance by the caller before interpretation.
    """
    task = 'temporal_pair_compatibility'
    labels = ('compatible', 'incompatible', 'insufficient_context')


class NegationScopeTokenHead(DenseTaskHead):
    """Per-token negation-scope candidate, not surface-word matching.

    Caller provides context features/offsets. No trained Arabic syntax or
    dialect handling is shipped. Membership is not claim refutation.
    """
    task = 'negation_scope_token_membership'
    labels = ('outside', 'inside_scope')


class EvidenceContradictionHead(DenseTaskHead):
    """Pairwise evidence relation; different entities/times may be unrelated.

    Scores alone never promote a candidate conflict to an editorial verdict.
    Requires independently reviewed event and context labels for acceptance.
    """
    task = 'evidence_pair_relation'
    labels = ('consistent', 'contradictory', 'insufficient_context')
