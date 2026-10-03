"""Experimental editorial review controls, never automatic truth verification.

These deterministic controls preserve raw Unicode code-point offsets and expose
review signals. They are not neural layers and do not produce truth scores.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .core import ValidationError


@dataclass(frozen=True)
class RawDocument:
    """A source document as collected, with its provenance and content hash."""
    source_id: str
    text: str
    source_url: str
    rights: str

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in (
            self.source_id, self.text, self.source_url, self.rights
        )):
            raise ValidationError('معرف ونص ورابط وبيان حقوق غير فارغة مطلوبة')

    @property
    def sha256(self) -> str:
        """Return the hex SHA-256 digest of the text."""
        return hashlib.sha256(self.text.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class EvidenceSpan:
    """A quoted span of a document used as evidence, with its offsets."""
    source_id: str
    source_sha256: str
    start: int
    end: int
    excerpt: str
    offset_unit: str = 'unicode_code_points'

    def verify(self, source: RawDocument) -> None:
        """Raise ValidationError unless the span matches the source document and its offsets."""
        if self.source_id != source.source_id or self.source_sha256 != source.sha256:
            raise ValidationError('المصدر أو نسخة المصدر لا يطابق الاقتباس')
        if (self.offset_unit != 'unicode_code_points' or
                type(self.start) is not int or type(self.end) is not int or
                not 0 <= self.start < self.end <= len(source.text) or
                source.text[self.start:self.end] != self.excerpt):
            raise ValidationError('الاقتباس أو مواضع الحروف غير مطابقة للنص الخام')


def exact_duplicate_groups(documents: Sequence[RawDocument]) -> tuple[tuple[str, ...], ...]:
    """Exact identical content is not an independent-source count.

    Different texts are not proof of independent origin. No semantic/event
    equivalence or copied-article attribution is inferred from this function.
    """
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for doc in documents:
        if doc.source_id in seen:
            raise ValidationError('معرف مصدر مكرر')
        seen.add(doc.source_id)
        groups.setdefault(doc.sha256, []).append(doc.source_id)
    return tuple(tuple(ids) for ids in groups.values() if len(ids) > 1)


_NUMBER = re.compile(r'(?<![\w])[-+]?\d+(?:[.,٫]\d+)*(?:%|٪)?')
_NEGATIONS = frozenset({'لا', 'لم', 'لن', 'ليس', 'ليست', 'غير', 'دون', 'مافيش', 'مش'})


def observed_numbers(text: str) -> frozenset[str]:
    """Surface numbers only; decimal/locale ambiguity remains visible."""
    values = set()
    for match in _NUMBER.finditer(text):
        token = unicodedata.normalize('NFKC', match.group())
        token = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in token)
        values.add(token.replace('٫', '.').replace('٪', '%'))
    return frozenset(values)


def comparison_signals(claim: str, excerpt: str) -> dict[str, Any]:
    """Diagnostics for editors. Missing signals never mean supported/true.

    Numerals can be dates, money, percentages or counts. This function does not
    align units, entities, times or denominators and never infers contradiction.
    """
    if not claim.strip() or not excerpt.strip():
        raise ValidationError('نص ادعاء واقتباس غير فارغين مطلوبان')
    cn, en = observed_numbers(claim), observed_numbers(excerpt)
    def negations(text: str) -> list[str]:
        """Return the negation cues found in the text."""
        tokens = set(re.findall(r'\w+', text, re.UNICODE))
        return sorted(tokens & _NEGATIONS)
    return {'claim_numbers': sorted(cn), 'excerpt_numbers': sorted(en),
            'claim_numbers_missing_from_excerpt': sorted(cn - en),
            'claim_negation_markers': negations(claim),
            'excerpt_negation_markers': negations(excerpt),
            'automatic_verdict': None, 'needs_editor_review': True}


_ALLOWED = frozenset({'needs_review', 'insufficient_evidence', 'conflicting_sources',
                      'editor_supported', 'editor_refuted'})


@dataclass(frozen=True)
class EditorialDecision:
    """The editor decision recorded for a claim; the library only suggests, it never decides."""
    claim: str
    state: str
    reviewer: str | None = None
    note: str = ''

    def __post_init__(self) -> None:
        if not self.claim.strip() or self.state not in _ALLOWED:
            raise ValidationError('ادعاء وحالة مراجعة معروفة مطلوبان')
        if self.state.startswith('editor_') and (
            not self.reviewer or not self.reviewer.strip() or not self.note.strip()
        ):
            raise ValidationError('قرار المحرر يحتاج معرف المراجع وسبب القرار')


def review_export(decision: EditorialDecision, documents: Sequence[RawDocument],
                  spans: Sequence[EvidenceSpan]) -> dict[str, Any]:
    """Export reviewed raw spans with provenance; not a trained model verdict."""
    sources = {doc.source_id: doc for doc in documents}
    if len(sources) != len(documents):
        raise ValidationError('معرف مصدر مكرر')
    if decision.state.startswith('editor_') and not spans:
        raise ValidationError('قرار محرر بدون اقتباس لا يصدر')
    output = []
    for span in spans:
        if span.source_id not in sources:
            raise ValidationError('مصدر الاقتباس غير موجود')
        doc = sources[span.source_id]
        span.verify(doc)
        output.append({'source_id': doc.source_id, 'source_url': doc.source_url,
                       'source_sha256': doc.sha256, 'rights': doc.rights,
                       'start': span.start, 'end': span.end, 'excerpt': span.excerpt,
                       'offset_unit': span.offset_unit})
    payload = {'format': 'newsrag-editorial-experimental-1', 'claim': decision.claim,
               'state': decision.state, 'reviewer': decision.reviewer,
               'note': decision.note, 'evidence': output,
               'automatic_truth_probability': None,
               'exact_duplicate_groups': exact_duplicate_groups(documents)}
    # Integrity checksum, not a signature or proof of reviewer identity.
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    payload['export_sha256'] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload
