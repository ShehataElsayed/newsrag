"""Experimental four-stage automatic triage pipeline (private lab code).

Stages run with no human step: (1) claim-type detection by plain rules,
(2) retrieval with similarity, (3) a contradiction check on retrieved
neighbours from any NLI model the caller supplies, (4) a decision that either
suggests a label or abstains, with reasons. The pipeline automates triage only.
Its output is a suggestion for an editor, never a published verdict: it carries
validated_for_release=False and autonomous use stays off until at least 500
independently reviewed cases support the thresholds. Claim types come from
rules, not a trained model, and their accuracy is not measured.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .core import ValidationError

_DIGITS = re.compile(r'[0-9\u0660-\u0669%]')
_ATTRIBUTION = re.compile(r'\u0642\u0627\u0644|\u0635\u0631\u062d|\u0635\u0631\u0651\u062d|\u0623\u0639\u0644\u0646|\u0627\u0639\u0644\u0646|\bsaid\b|\bclaimed\b')
_HEALTH = re.compile(r'\u0644\u0642\u0627\u062d|\u0641\u064a\u0631\u0648\u0633|\u0643\u0648\u0631\u0648\u0646\u0627|\u0639\u0644\u0627\u062c|\u0645\u0631\u0636|\u0635\u062d\u0629|vaccine|virus|cancer|disease')
CLAIM_TYPES = ('numeric', 'attribution', 'health', 'general')


def claim_type(text: str) -> str:
    """Rule-based type: health cues first, then attribution, then numbers, else general."""
    if _HEALTH.search(text):
        return 'health'
    if _ATTRIBUTION.search(text):
        return 'attribution'
    if _DIGITS.search(text):
        return 'numeric'
    return 'general'


@dataclass(frozen=True)
class PipelineConfig:
    """Thresholds, fixed before looking at accuracy."""

    min_similarity: float = 0.80
    contradiction_threshold: float = 0.5
    min_vote_share: float = 0.6
    skip_types: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Decision:
    """One pipeline outcome: a suggestion or an abstention with reasons."""

    claim_type: str
    suggestion: str | None
    abstain: bool
    reasons: tuple[str, ...]
    top_similarity: float
    max_contradiction: float
    validated_for_release: bool = False
    autonomous_allowed: bool = False
    status: str = 'suggestion for an editor; experimental'


def decide(kind: str, neighbour_labels: Sequence[str], similarities: Sequence[float],
           contradictions: Sequence[float], config: PipelineConfig | None = None,
           *, stages: int = 4) -> Decision:
    """Stage 4 decision. `stages` (2 to 4) enables ablations.

    stages=2: similarity only. 3: plus contradiction abstention. 4: plus split-vote
    and skipped-type abstention. Contradiction from any near neighbour abstains.
    """
    config = config or PipelineConfig()
    n = len(neighbour_labels)
    if n != len(similarities) or n != len(contradictions):
        raise ValidationError('neighbour labels, similarities and contradictions must align')
    if stages not in (2, 3, 4):
        raise ValidationError('stages must be 2, 3 or 4')
    top = max(similarities, default=0.0)
    worst = max(contradictions, default=0.0)

    def out(reasons: tuple[str, ...], suggestion: str | None = None) -> Decision:
        return Decision(kind, suggestion, suggestion is None, reasons, top, worst)

    if stages >= 4 and kind in config.skip_types:
        return out(('skipped_type',))
    near = [i for i in range(n) if similarities[i] >= config.min_similarity]
    if not near:
        return out(('low_similarity',))
    if stages >= 3 and any(contradictions[i] >= config.contradiction_threshold for i in near):
        return out(('contradiction',))
    if stages == 2:
        return out((), neighbour_labels[near[0]])
    votes: dict[str, float] = {}
    for i in near:
        votes[neighbour_labels[i]] = votes.get(neighbour_labels[i], 0.0) + similarities[i]
    label, weight = max(votes.items(), key=lambda kv: kv[1])
    if weight / sum(votes.values()) < config.min_vote_share:
        return out(('split_vote',))
    return out((), label)

