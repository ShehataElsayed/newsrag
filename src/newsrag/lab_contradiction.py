"""Experimental contradiction gate (private lab code).

A retrieved passage can be close in embedding space and still say the opposite
of the claim. This module turns pair probabilities from any natural-language
inference model (the caller supplies them; none is shipped) into an abstention
flag, and reports how accurate the kept and flagged groups are. It is a second
layer beside cosine similarity. It does not validate anything.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .core import ValidationError
from .lab_stats import Interval, wilson_interval


def contradiction_gate(contradiction: Sequence[float], *, threshold: float = 0.5) -> list[bool]:
    """True where the contradiction probability reaches the threshold (abstain)."""
    if not 0.0 < threshold <= 1.0:
        raise ValidationError('threshold must be in (0, 1]')
    if any(p < 0.0 or p > 1.0 for p in contradiction):
        raise ValidationError('probabilities must be in [0, 1]')
    return [p >= threshold for p in contradiction]


@dataclass(frozen=True)
class GateReport:
    """Effect of abstaining on flagged pairs."""

    total: int
    flagged: int
    coverage: float
    accuracy_all: Interval
    accuracy_kept: Interval | None
    accuracy_flagged: Interval | None
    status: str = 'experimental; validated_for_release=False'


def gate_report(flags: Sequence[bool], correct: Sequence[bool]) -> GateReport:
    """Accuracy of all, kept and flagged items. `correct` says the retrieved evidence agreed."""
    if len(flags) != len(correct) or not flags:
        raise ValidationError('flags and correct must align and be non-empty')
    kept = [c for f, c in zip(flags, correct, strict=True) if not f]
    cut = [c for f, c in zip(flags, correct, strict=True) if f]
    return GateReport(
        len(flags), len(cut), len(kept) / len(flags),
        wilson_interval(sum(correct), len(correct)),
        wilson_interval(sum(kept), len(kept)) if kept else None,
        wilson_interval(sum(cut), len(cut)) if cut else None)
