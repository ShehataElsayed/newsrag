"""Small statistics helpers for honest evaluation; none of them validate a model.

A significant difference on weak labels is still a difference on weak labels.
Calibration here describes agreement on a labeled set, not truth probabilities.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .core import ValidationError
from .neural import _numpy


@dataclass(frozen=True)
class Interval:
    """A point estimate with a lower and upper bound."""
    estimate: float
    low: float
    high: float
    cases: int


def bootstrap_interval(values: Sequence[float], *, resamples: int = 2000, seed: int = 7,
                       level: float = 0.95) -> Interval:
    """Percentile bootstrap of the mean over cases."""
    if len(values) < 2 or resamples < 100 or not 0.5 < level < 1:
        raise ValidationError('قيمتان على الأقل وإعادات كافية ومستوى ثقة صالح مطلوبة')
    np = _numpy()
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).all():
        raise ValidationError('قيم محدودة مطلوبة')
    means = v[np.random.default_rng(seed).integers(0, len(v), size=(resamples, len(v)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return Interval(float(v.mean()), float(np.percentile(means, tail)),
                    float(np.percentile(means, 100 - tail)), len(v))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Interval:
    """Wilson score interval for a proportion (successes out of n)."""
    if total < 1 or not 0 <= successes <= total:
        raise ValidationError('عدد نجاحات وإجمالي صالحان مطلوبان')
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), total)


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, float]:
    """Exact two-sided McNemar test on paired correctness; b01/b10 are discordant counts."""
    if len(correct_a) != len(correct_b) or not correct_a:
        raise ValidationError('نتائج مزدوجة متساوية وغير فارغة مطلوبة')
    only_a = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if a and not b)
    only_b = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if b and not a)
    n = only_a + only_b
    if n == 0:
        return {'only_a': 0, 'only_b': 0, 'p_value': 1.0}
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {'only_a': only_a, 'only_b': only_b, 'p_value': min(1.0, 2 * tail)}


def paired_permutation_pvalue(a: Sequence[float], b: Sequence[float], *, resamples: int = 10000,
                              seed: int = 7) -> float:
    """Two-sided sign-flip permutation test on the mean paired difference."""
    if len(a) != len(b) or len(a) < 2 or resamples < 1000:
        raise ValidationError('قياسات مزدوجة متساوية وإعادات كافية مطلوبة')
    np = _numpy()
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(float(d.mean()))
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(resamples, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return float((1 + (perm >= observed - 1e-12).sum()) / (resamples + 1))


def minimum_detectable_difference(n: int, discordant_rate: float, *, z_alpha: float = 1.96,
                                  z_beta: float = 0.84) -> float:
    """Approximate paired accuracy difference detectable with ~80% power.

    Uses the normal approximation of McNemar: diff ~ (z_a+z_b)*sqrt(discordant_rate/n)."""
    if n < 2 or not 0 < discordant_rate <= 1:
        raise ValidationError('عدد حالات ونسبة تباين صالحان مطلوبان')
    return (z_alpha + z_beta) * math.sqrt(discordant_rate / n)


@dataclass(frozen=True)
class Calibration:
    """Expected calibration error, Brier score and the reliability bins."""
    brier: float
    ece: float
    bins: tuple[tuple[float, float, int], ...]  # mean confidence, accuracy, count


def calibration(confidences: Sequence[float], correct: Sequence[bool], *, bins: int = 10
                ) -> Calibration:
    """Brier score and expected calibration error over equal-width confidence bins."""
    if len(confidences) != len(correct) or not confidences or bins < 2:
        raise ValidationError('ثقات ونتائج بعدد متساو وفئات كافية مطلوبة')
    if any(not (0.0 <= c <= 1.0) or not math.isfinite(c) for c in confidences):
        raise ValidationError('ثقة ضمن 0 و1 مطلوبة')
    n = len(confidences)
    brier = sum((c - float(o)) ** 2 for c, o in zip(confidences, correct, strict=True)) / n
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for c, o in zip(confidences, correct, strict=True):
        grouped[min(int(c * bins), bins - 1)].append((c, o))
    out = []
    ece = 0.0
    for g in grouped:
        if g:
            conf = sum(c for c, _ in g) / len(g)
            acc = sum(o for _, o in g) / len(g)
            ece += len(g) / n * abs(conf - acc)
            out.append((conf, acc, len(g)))
    return Calibration(brier, ece, tuple(out))


def error_breakdown(groups: Sequence[str], correct: Sequence[bool]) -> dict[str, dict[str, float]]:
    """Accuracy and count per group, for finding where a model fails."""
    if len(groups) != len(correct) or not groups:
        raise ValidationError('مجموعات ونتائج بعدد متساو وغير فارغة مطلوبة')
    totals, hits = Counter(groups), Counter(g for g, o in zip(groups, correct, strict=True) if o)
    return {g: {'count': totals[g], 'accuracy': hits[g] / totals[g]} for g in sorted(totals)}
