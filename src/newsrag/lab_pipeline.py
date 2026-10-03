"""Experimental review workspace, raw span decoder and provenance graph.

No web server, authentication, storage service or automatic truth verdict.
Reviewed edges encode caller assertions; they do not authenticate reviewers.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .core import ValidationError
from .lab import (
    EditorialDecision,
    EvidenceSpan,
    RawDocument,
    comparison_signals,
    review_export,
)


@dataclass(frozen=True)
class TokenOffset:
    """Raw Unicode code-point offsets, never decoded/normalized text offsets."""
    start: int
    end: int
    raw_text: str

    def verify(self, document: RawDocument) -> None:
        """Raise ValidationError unless the token offsets match the document text."""
        if (type(self.start) is not int or type(self.end) is not int
                or not 0 <= self.start < self.end <= len(document.text)
                or document.text[self.start:self.end] != self.raw_text):
            raise ValidationError('خريطة الرموز لا تطابق النص الخام')


def decode_evidence_spans(document: RawDocument, offsets: Sequence[TokenOffset],
                          inside_scores: Sequence[float], *, threshold: float,
                          max_chars: int = 2000) -> tuple[EvidenceSpan, ...]:
    """Join consecutive selected tokens only across whitespace gaps.

    Threshold is caller-selected, not calibrated here. All offsets are checked,
    including tokens below threshold. Non-whitespace gaps split spans so an
    omitted clause cannot be silently included. Output is candidate evidence,
    not reviewed evidence. Only supplied token coverage can be decoded.
    """
    if (len(offsets) != len(inside_scores) or not math.isfinite(threshold)
            or not 0 <= threshold <= 1 or type(max_chars) is not int or max_chars < 1):
        raise ValidationError('عتبة وطول وحدود رموز متسقة مطلوبة')
    previous_end = 0
    for token, score in zip(offsets, inside_scores, strict=True):
        token.verify(document)
        if token.start < previous_end:
            raise ValidationError('رموز متداخلة أو غير مرتبة')
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValidationError('درجة عضوية رمز غير صالحة')
        previous_end = token.end
    spans: list[EvidenceSpan] = []
    start: int | None = None
    end = 0

    def finish() -> None:
        """Emit the span being built, if any, after verifying it against the document."""
        nonlocal start
        if start is not None:
            span = EvidenceSpan(document.source_id, document.sha256, start, end,
                                document.text[start:end])
            span.verify(document)
            spans.append(span)
            start = None

    for token, score in zip(offsets, inside_scores, strict=True):
        if score < threshold:
            finish()
            continue
        if token.end - token.start > max_chars:
            raise ValidationError('رمز يتجاوز أقصى طول اقتباس')
        if start is not None and (document.text[end:token.start].strip()
                                  or token.end-start > max_chars):
            finish()
        if start is None:
            start = token.start
        end = token.end
    finish()
    return tuple(spans)


@dataclass(frozen=True)
class ProvenanceEdge:
    """A documented editorial assertion, not model-generated attribution."""
    citing_source_id: str
    origin_source_id: str
    relation: str
    reviewer: str
    note: str
    evidence: EvidenceSpan

    def __post_init__(self) -> None:
        if (any(not isinstance(x, str) or not x.strip() for x in (
            self.citing_source_id, self.origin_source_id, self.reviewer, self.note))
                or self.citing_source_id == self.origin_source_id
                or self.relation not in {'cites', 'copies', 'derived_from'}):
            raise ValidationError('علاقة أصل بمراجع وسبب ومصدرين مختلفين مطلوبة')


class ProvenanceGraph:
    """Reviewed directed source-origin graph, with cycles visible, not hidden.

    Absence of an edge never proves independence. Connected groups are known
    link groups, not an independent-source count. Reviewer identity is supplied
    text and not authenticated. Cycles can exist in citation networks.
    """
    def __init__(self, documents: Sequence[RawDocument]):
        self.documents = {d.source_id: d for d in documents}
        if len(self.documents) != len(documents):
            raise ValidationError('معرف مصدر مكرر')
        self.edges: list[ProvenanceEdge] = []

    def add(self, edge: ProvenanceEdge) -> None:
        """Add a provenance edge after verifying its evidence; duplicates are rejected."""
        if (edge.citing_source_id not in self.documents
                or edge.origin_source_id not in self.documents
                or edge.evidence.source_id != edge.citing_source_id):
            raise ValidationError('المصدر أو دليل علاقة النقل غير موجود')
        edge.evidence.verify(self.documents[edge.citing_source_id])
        if edge in self.edges:
            raise ValidationError('علاقة أصل مكررة')
        self.edges.append(edge)

    def origin_path(self, source_id: str) -> tuple[str, ...]:
        """All transitively reachable origins, cycle safe; excludes self."""
        if source_id not in self.documents:
            raise ValidationError('مصدر غير موجود')
        seen = {source_id}
        pending = [source_id]
        while pending:
            current = pending.pop()
            for edge in self.edges:
                if edge.citing_source_id == current and edge.origin_source_id not in seen:
                    seen.add(edge.origin_source_id)
                    pending.append(edge.origin_source_id)
        return tuple(sorted(seen - {source_id}))

    def audit(self) -> dict[str, Any]:
        """Summarise the provenance graph: edge count, cycles, and unset independence fields."""
        return {'edge_count': len(self.edges),
                'cycles_present': any(e.citing_source_id in self.origin_path(e.origin_source_id)
                                      for e in self.edges),
                'independent_source_count': None,
                'reviewer_identity_authenticated': False,
                'edges': [{'citing_source_id': e.citing_source_id,
                           'origin_source_id': e.origin_source_id, 'relation': e.relation,
                           'reviewer': e.reviewer, 'note': e.note,
                           'evidence_source_sha256': e.evidence.source_sha256,
                           'evidence_start': e.evidence.start, 'evidence_end': e.evidence.end,
                           'evidence_excerpt': e.evidence.excerpt} for e in self.edges]}


@dataclass
class ReviewWorkspace:
    """Local API surface joining candidates, raw sources and editorial export.

    No trained model is silently selected. Neural heads are separate experimental
    dependencies; callers may add only offset-verified candidate spans here.
    An explicit EditorialDecision is required to finish, never a neural score.
    """
    claim: str
    _documents: dict[str, RawDocument] = field(default_factory=dict, init=False, repr=False)
    _candidates: list[EvidenceSpan] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.claim, str) or not self.claim.strip():
            raise ValidationError('نص ادعاء غير فارغ مطلوب')

    def add_document(self, document: RawDocument) -> None:
        """Register a source document; a repeated source id is rejected."""
        if document.source_id in self._documents:
            raise ValidationError('نسخة المصدر موجودة؛ أنشئ مراجعة جديدة لتحديثها')
        self._documents[document.source_id] = document

    def add_candidate(self, span: EvidenceSpan) -> None:
        """Register a candidate evidence span after verifying it against its source."""
        if span.source_id not in self._documents:
            raise ValidationError('أضف نسخة المصدر قبل الاقتباس')
        span.verify(self._documents[span.source_id])
        if span in self._candidates:
            raise ValidationError('اقتراح اقتباس مكرر')
        self._candidates.append(span)

    def diagnostics(self) -> tuple[dict[str, Any], ...]:
        """Return the comparison signals for every candidate span (numbers, negations); no verdict."""
        return tuple({'source_id': span.source_id, 'start': span.start, 'end': span.end,
                      'excerpt': span.excerpt,
                      **comparison_signals(self.claim, span.excerpt)}
                     for span in self._candidates)

    def finish(self, decision: EditorialDecision, selected_indices: Sequence[int]) -> dict[str, Any]:
        """Record the editor decision and the selected candidate spans and return the review record."""
        if decision.claim != self.claim:
            raise ValidationError('قرار المراجعة يخص ادعاء مختلفا')
        if (any(type(i) is not int or not 0 <= i < len(self._candidates)
                for i in selected_indices) or len(set(selected_indices)) != len(selected_indices)):
            raise ValidationError('اختيار اقتباسات غير صالح')
        return review_export(decision, tuple(self._documents.values()),
                             tuple(self._candidates[i] for i in selected_indices))
