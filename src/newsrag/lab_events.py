"""Experimental event grouping and leak-impact evaluation (private lab code).

Claims about the same event often share wording. A random or purely temporal
split can then put near-duplicates on both sides and inflate accuracy. This
module groups near-duplicate claims, builds group-disjoint splits, and measures
how much accuracy depends on having a close neighbour in training.

Groups are a lexical proxy (cosine over hashed character n-grams), not a
verified event identity. Nothing here verifies claims.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .core import ValidationError
from .lab_stats import Interval, wilson_interval
from .lab_verdict import HashedTfidf
from .neural import _numpy


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def event_groups(texts: Sequence[str], *, threshold: float = 0.6, dim: int = 4096,
                 block: int = 512) -> list[int]:
    """Group texts into connected components of pairs with cosine >= threshold.

    Returns one group id per text (ids are consecutive from 0, ordered by first
    appearance). The comparison is blocked so memory stays bounded.
    """
    if not texts:
        raise ValidationError('texts are required')
    if not 0 < threshold <= 1:
        raise ValidationError('threshold must be in (0, 1]')
    np = _numpy()
    x = HashedTfidf(dim).fit(texts).transform(texts)
    parent = list(range(len(texts)))
    for start in range(0, len(texts), block):
        sims = x[start:start + block] @ x.T
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            i = start + r
            if i != c:
                parent[_find(parent, i)] = _find(parent, c)
    ids: dict[int, int] = {}
    return [ids.setdefault(_find(parent, i), len(ids)) for i in range(len(texts))]


def group_kfold(groups: Sequence[int], folds: int = 5, seed: int = 7) -> list[int]:
    """Assign each item a fold so that no group spans two folds (balanced by size)."""
    if folds < 2:
        raise ValidationError('at least 2 folds are required')
    np = _numpy()
    sizes: dict[int, int] = {}
    for g in groups:
        sizes[g] = sizes.get(g, 0) + 1
    order = list(sizes)
    np.random.default_rng(seed).shuffle(order)
    order.sort(key=lambda g: -sizes[g])
    load = [0] * folds
    where: dict[int, int] = {}
    for g in order:
        f = min(range(folds), key=load.__getitem__)
        where[g] = f
        load[f] += sizes[g]
    return [where[g] for g in groups]


def random_kfold(count: int, folds: int = 5, seed: int = 7) -> list[int]:
    """Plain random fold assignment, the leaky baseline."""
    if folds < 2 or count < folds:
        raise ValidationError('invalid fold configuration')
    np = _numpy()
    order = np.random.default_rng(seed).permutation(count)
    out = [0] * count
    for pos, i in enumerate(order.tolist()):
        out[i] = pos % folds
    return out


def nearest_train_similarity(train_texts: Sequence[str], test_texts: Sequence[str], *,
                             dim: int = 4096, block: int = 512) -> list[float]:
    """Highest cosine similarity of each test text to any training text."""
    if not train_texts or not test_texts:
        raise ValidationError('train and test texts are required')
    vec = HashedTfidf(dim).fit(list(train_texts) + list(test_texts))
    a, b = vec.transform(train_texts), vec.transform(test_texts)
    out: list[float] = []
    for start in range(0, len(test_texts), block):
        out.extend((b[start:start + block] @ a.T).max(axis=1).tolist())
    return out


@dataclass(frozen=True)
class LeakImpact:
    """Accuracy on test cases with and without a close training neighbour."""

    threshold: float
    near: Interval
    near_count: int
    novel: Interval
    novel_count: int

    @property
    def gap(self) -> float:
        """Accuracy on near-neighbour cases minus accuracy on novel cases."""
        return self.near.estimate - self.novel.estimate


def leak_impact(similarity: Sequence[float], correct: Sequence[bool], *,
                threshold: float = 0.6) -> LeakImpact:
    """Split test cases by nearest-training similarity and compare accuracy."""
    if len(similarity) != len(correct) or not similarity:
        raise ValidationError('similarity and correctness must align')
    near = [bool(c) for s, c in zip(similarity, correct, strict=True) if s >= threshold]
    novel = [bool(c) for s, c in zip(similarity, correct, strict=True) if s < threshold]
    if not near or not novel:
        raise ValidationError('both near and novel subsets must be non-empty')
    return LeakImpact(threshold, wilson_interval(sum(near), len(near)), len(near),
                      wilson_interval(sum(novel), len(novel)), len(novel))


def split_overlap(groups: Sequence[int], assignment: Sequence[Any]) -> int:
    """Number of groups that appear in more than one fold or side (0 = clean)."""
    seen: dict[int, set[Any]] = {}
    for g, a in zip(groups, assignment, strict=True):
        seen.setdefault(g, set()).add(a)
    return sum(len(v) > 1 for v in seen.values())


def event_groups_from_vectors(vectors: Any, *, threshold: float = 0.8, block: int = 512) -> list[int]:
    """Group rows of an embedding matrix by cosine similarity (semantic grouping).

    Pass any sentence-embedding matrix (rows are items). Rows are L2-normalised
    here. The embedder is the caller's choice; this module ships none, so the
    quality of the grouping is only as good as the embedder supplied.
    """
    np = _numpy()
    x = np.asarray(vectors, dtype=float)
    if x.ndim != 2 or not len(x):
        raise ValidationError('vectors must be a non-empty 2-D matrix')
    if not 0 < threshold <= 1:
        raise ValidationError('threshold must be in (0, 1]')
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    parent = list(range(len(x)))
    for start in range(0, len(x), block):
        rows, cols = np.nonzero(x[start:start + block] @ x.T >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            if start + r != c:
                parent[_find(parent, start + r)] = _find(parent, c)
    ids: dict[int, int] = {}
    return [ids.setdefault(_find(parent, i), len(ids)) for i in range(len(x))]


def _tokens(text: str) -> set[str]:
    return {t for t in ''.join(ch if ch.isalnum() else ' ' for ch in text.lower()).split()
            if len(t) >= 3}


def event_groups_shared_terms(texts: Sequence[str], days: Sequence[int], *, min_shared: int = 3,
                              window_days: int = 14, max_doc_freq: float = 0.01) -> list[int]:
    """Group claims that share several rare terms within a short date window.

    A rare term appears in at most `max_doc_freq` of the texts. Two claims join
    when they share at least `min_shared` rare terms and their dates are within
    `window_days`. This catches same-event claims worded differently when names,
    places or numbers repeat. It is still a proxy: it can merge unrelated claims
    and miss paraphrases with no shared terms.
    """
    if len(texts) != len(days) or not texts:
        raise ValidationError('texts and days must align and be non-empty')
    toks = [_tokens(t) for t in texts]
    freq: dict[str, int] = {}
    for s in toks:
        for t in s:
            freq[t] = freq.get(t, 0) + 1
    limit = max(2, int(max_doc_freq * len(texts)))
    rare = [{t for t in s if 2 <= freq[t] <= limit} for s in toks]
    postings: dict[str, list[int]] = {}
    for i, s in enumerate(rare):
        for t in s:
            postings.setdefault(t, []).append(i)
    parent = list(range(len(texts)))
    for i, s in enumerate(rare):
        counts: dict[int, int] = {}
        for t in s:
            for j in postings[t]:
                if j > i:
                    counts[j] = counts.get(j, 0) + 1
        for j, n in counts.items():
            if n >= min_shared and abs(days[i] - days[j]) <= window_days:
                parent[_find(parent, i)] = _find(parent, j)
    ids: dict[int, int] = {}
    return [ids.setdefault(_find(parent, i), len(ids)) for i in range(len(texts))]
