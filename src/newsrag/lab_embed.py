"""Experimental embedder ablation helpers (private lab code).

Scores any embedding matrix by nearest-training-neighbour label transfer on a
temporal split, next to the majority baseline. The embedder is the caller's
choice; none is shipped. Nothing here validates a model for release.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .core import ValidationError
from .lab_stats import Interval, wilson_interval


@dataclass(frozen=True)
class TransferResult:
    """Nearest-neighbour label transfer for one embedder."""

    accuracy: Interval
    majority: Interval
    similarity: list[float]
    neighbour: list[int]
    hits: list[bool]
    status: str = 'experimental; validated_for_release=False'


def nearest_label_transfer(train: Any, train_labels: Sequence[str], test: Any,
                           test_labels: Sequence[str], *, block: int = 512) -> TransferResult:
    """Label each test row with its most similar training row's label (rows must be L2-normalised)."""
    import numpy as np

    if len(train) != len(train_labels) or len(test) != len(test_labels) or not len(train) or not len(test):
        raise ValidationError('embeddings and labels must align and be non-empty')
    sims: list[float] = []
    nbr: list[int] = []
    tr = np.asarray(train, dtype=np.float32)
    for s in range(0, len(test), block):
        m = np.asarray(test[s:s + block], dtype=np.float32) @ tr.T
        nbr.extend(int(i) for i in m.argmax(axis=1))
        sims.extend(float(v) for v in m.max(axis=1))
    hits = [train_labels[j] == y for j, y in zip(nbr, test_labels, strict=True)]
    top = Counter(train_labels).most_common(1)[0][0]
    maj = [y == top for y in test_labels]
    return TransferResult(wilson_interval(sum(hits), len(hits)), wilson_interval(sum(maj), len(maj)),
                          sims, nbr, hits)
