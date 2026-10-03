"""Experimental agreement gate (private lab code).

Predicts whether a retrieved neighbour's label will match the claim's label,
from pair features the caller builds (similarities, NLI probabilities, label
and type indicators). A plain numpy logistic regression with standardised
features and L2. It is a learned replacement for a single contradiction cut-off.
The target is label agreement, not logical contradiction, and nothing here
validates a model for release.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .core import ValidationError
from .neural import _numpy


def auroc(scores: Sequence[float], positives: Sequence[bool]) -> float:
    """Area under the ROC curve by rank (ties share the average rank)."""
    if len(scores) != len(positives):
        raise ValidationError('scores and positives must align')
    pos = sum(1 for p in positives if p)
    neg = len(positives) - pos
    if pos == 0 or neg == 0:
        raise ValidationError('need both positive and negative cases')
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, p in zip(ranks, positives, strict=True) if p)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


@dataclass(frozen=True)
class AgreementGate:
    """A fitted logistic model over standardised pair features."""

    weights: Any
    bias: float
    mean: Any
    scale: Any
    status: str = 'experimental; validated_for_release=False'

    def probability(self, features: Any) -> Any:
        """Probability that the neighbour's label agrees, one value per row."""
        np = _numpy()
        x = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        z = np.clip(x @ self.weights + self.bias, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))


def fit_agreement_gate(features: Any, agrees: Sequence[bool], *, l2: float = 1e-2,
                       epochs: int = 400, lr: float = 0.3) -> AgreementGate:
    """Fit by full-batch gradient descent. Rows are pairs, columns are features."""
    np = _numpy()
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(agrees, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or len(y) < 10:
        raise ValidationError('features must be 2-D, align with labels, and have at least 10 rows')
    if not 0.0 < float(y.mean()) < 1.0:
        raise ValidationError('need both agreeing and disagreeing pairs')
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    w = np.zeros(z.shape[1])
    b = math.log(float(y.mean()) / (1.0 - float(y.mean())))
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ w + b, -30.0, 30.0)))
        err = p - y
        w -= lr * (z.T @ err / len(y) + l2 * w)
        b -= lr * float(err.mean())
    return AgreementGate(w, b, mean, scale)
