"""Experimental drift monitor for label and score distributions (private lab code).

The verdict experiments showed the label mix shifting over time, which makes
accuracy numbers from different periods hard to compare. This module measures
the shift so it can be reported next to any accuracy figure. It does not
correct for drift and does not validate anything.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .core import ValidationError


@dataclass(frozen=True)
class DriftReport:
    """Shift between a reference period and a later period."""

    psi: float
    total_variation: float
    reference: dict[str, float]
    current: dict[str, float]
    status: str = 'experimental; validated_for_release=False'

    @property
    def level(self) -> str:
        """Conventional PSI reading: below 0.1 small, below 0.25 moderate, else large."""
        return 'small' if self.psi < 0.1 else 'moderate' if self.psi < 0.25 else 'large'


def _shares(labels: Sequence[str], classes: Sequence[str], smoothing: float) -> dict[str, float]:
    counts = Counter(labels)
    total = len(labels) + smoothing * len(classes)
    return {c: (counts.get(c, 0) + smoothing) / total for c in classes}


def label_drift(reference: Sequence[str], current: Sequence[str], classes: Sequence[str], *,
                smoothing: float = 0.5) -> DriftReport:
    """Population stability index and total variation distance between two label sets."""
    if not reference or not current:
        raise ValidationError('reference and current labels are required')
    if len(set(classes)) != len(classes) or len(classes) < 2:
        raise ValidationError('classes must be unique and at least two')
    if smoothing <= 0:
        raise ValidationError('smoothing must be positive')
    ref = _shares(reference, classes, smoothing)
    cur = _shares(current, classes, smoothing)
    psi = sum((cur[c] - ref[c]) * math.log(cur[c] / ref[c]) for c in classes)
    tv = 0.5 * sum(abs(cur[c] - ref[c]) for c in classes)
    return DriftReport(psi, tv, ref, cur)
