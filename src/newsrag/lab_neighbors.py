"""Experimental near-neighbour gap report (private lab code).

Compares accuracy on test items that have a close training neighbour with
accuracy on novel items, and also reports the gap after standardising the near
group to the novel group's label mix. A gap that shrinks after standardising is
mostly label mix; a gap that stays points to duplicate buoyancy. Neither proves
event independence. It does not validate anything.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .core import ValidationError
from .lab_stats import Interval, wilson_interval


@dataclass(frozen=True)
class NeighbourGap:
    """Near versus novel accuracy at one similarity threshold."""

    threshold: float
    near_n: int
    novel_n: int
    near_accuracy: Interval
    novel_accuracy: Interval
    raw_gap: float
    standardised_gap: float | None
    coverage: float
    status: str = 'experimental; validated_for_release=False'


def _acc(correct: Sequence[bool]) -> Interval:
    return wilson_interval(sum(correct), len(correct))


def neighbour_gap(similarity: Sequence[float], correct: Sequence[bool], labels: Sequence[str],
                  threshold: float) -> NeighbourGap:
    """Gap between items whose best training similarity is >= threshold and the rest.

    Standardising uses only classes present in both groups, with the novel
    group's class weights renormalised; coverage is the share of novel items in
    those classes. standardised_gap is None when no class is shared.
    """
    if not len(similarity) == len(correct) == len(labels):
        raise ValidationError('similarity, correct and labels must have the same length')
    near = [i for i, s in enumerate(similarity) if s >= threshold]
    novel = [i for i, s in enumerate(similarity) if s < threshold]
    if not near or not novel:
        raise ValidationError('threshold leaves one group empty')
    near_acc, novel_acc = _acc([correct[i] for i in near]), _acc([correct[i] for i in novel])
    novel_labels = [labels[i] for i in novel]
    mix = Counter(novel_labels)
    near_by: dict[str, list[bool]] = {}
    for i in near:
        near_by.setdefault(labels[i], []).append(correct[i])
    shared = [c for c in mix if c in near_by]
    covered = sum(mix[c] for c in shared)
    if not shared:
        return NeighbourGap(threshold, len(near), len(novel), near_acc, novel_acc,
                            near_acc.estimate - novel_acc.estimate, None, 0.0)
    std = sum(mix[c] / covered * (sum(near_by[c]) / len(near_by[c])) for c in shared)
    novel_shared = sum(correct[i] for i in novel if labels[i] in near_by) / covered
    return NeighbourGap(threshold, len(near), len(novel), near_acc, novel_acc,
                        near_acc.estimate - novel_acc.estimate, std - novel_shared, covered / len(novel))
