"""Small, deterministic retrieval evaluation; not an answer-truth benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from .core import NewsroomRAG, ValidationError


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_source_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.query.strip() or not self.relevant_source_ids or any(
            not source_id.strip() for source_id in self.relevant_source_ids
        ):
            raise ValidationError("An evaluation case needs a query and relevant source IDs")


@dataclass(frozen=True)
class RetrievalScore:
    cases: int
    top_k: int
    recall_at_k: float
    hit_rate_at_k: float
    reciprocal_rank: float
    missed_queries: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "top_k": self.top_k,
            "recall_at_k": self.recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "missed_queries": list(self.missed_queries),
        }


def evaluate_retrieval(rag: NewsroomRAG, cases: list[RetrievalCase], *,
                       top_k: int = 5) -> RetrievalScore:
    """Score retrieved distinct source IDs against hand-labeled relevant IDs.

    Rerankers and embedding providers configured on `rag` run as in ordinary search.
    Cases must come from editorial labels independent of the retrieved ranking.
    """
    if not cases or top_k < 1:
        raise ValidationError("At least one case and top_k >= 1 required")
    recalls = []
    hit_rates = []
    reciprocal_ranks = []
    missed = []
    for case in cases:
        hits = rag.search(case.query, top_k=top_k, one_per_source=True)
        ids = [hit.source_id for hit in hits]
        relevant = case.relevant_source_ids
        recalls.append(len(set(ids) & relevant) / len(relevant))
        ranks = [rank for rank, source_id in enumerate(ids, 1) if source_id in relevant]
        hit_rates.append(float(bool(ranks)))
        reciprocal_ranks.append(1 / ranks[0] if ranks else 0.0)
        if not ranks:
            missed.append(case.query)
    return RetrievalScore(len(cases), top_k, mean(recalls), mean(hit_rates),
                          mean(reciprocal_ranks), tuple(missed))
