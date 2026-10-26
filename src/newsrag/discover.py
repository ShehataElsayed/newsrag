"""Discovery stage: research question -> WHICH declared sources to collect.

Scope is deliberately bounded: the library never crawls the open web and
Wikipedia is excluded everywhere. Discovery ranks the DECLARED source
registry (each entry carries publisher, topics and reference-period metadata)
against the question. Selection is deterministic keyword/topic overlap first;
an optional local-model pass may RE-RANK but never ADD a source that scored
zero - the model can only order what the deterministic pass already admitted.
Sources the question gives no signal for land in `unselected` with reasons,
so the gap is visible instead of silently dropped.
"""
from __future__ import annotations

import re
from typing import Any, Callable

_WORD = re.compile(r"[A-Za-zء-ي]{3,}")
_STOP = set("""ما ماهو ماهي كم هل في من على إلى عن أن إن كان كانت هذا هذه ذلك التي الذي
the and for with what which how many much did does are was were has have had
سنة شهر عام سبتمبر أغسطس اكتوبر أكتوبر""".split())


# Bounded bilingual glossary for AR<->EN scope matching: the library takes
# Arabic questions over Arabic AND English sources; these aliases canonicalize
# the case domain's recurring proper nouns/topic words to one token. Bounded
# and documented - not per-trial tuning; extend only with domain terms.
_ALIASES = {}
for canon, variants in {
    "egypt": ("egypt", "egyptian", "egyptians", "مصر", "مصري", "مصريين", "المصريين"),
    "saudi": ("saudi", "saudis", "السعودية", "سعودي", "سعودية", "سعوديين"),
    "campaign": ("campaign", "campaigns", "حملة", "حملات"),
    "coordinated": ("coordinated", "coordination", "منسقة", "منظم", "منظمة"),
    "deport": ("deport", "deportation", "ترحيل"),
    "boycott": ("boycott", "مقاطعة"),
    "investigation": ("investigation", "investigations", "تحقيق", "تحقيقات"),
    "inflation": ("inflation", "تضخم"),
    "population": ("population", "سكان", "السكان"),
    "gdp": ("gdp", "الناتج", "ناتج"),
    "unemployment": ("unemployment", "بطالة", "البطالة"),
    "growth": ("growth", "نمو", "النمو"),
    "rate": ("rate", "معدل", "سعر", "المعدل"),
    "prices": ("price", "prices", "أسعار", "اسعار"),
}.items():
    for v in variants:
        _ALIASES[v] = canon


def _stem(tok: str) -> str:
    """Light Arabic normalization: definite article, common clitics and plural
    endings, so التضخم matches تضخم and الحضر matches حضر. Not a morphological
    analyzer - good enough for declared-registry keyword matching."""
    for pre in ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ل", "ف"):
        if tok.startswith(pre) and len(tok) - len(pre) >= 3:
            tok = tok[len(pre):]
            break
    for suf in ("يات", "ات", "ين", "ون", "ة", "ه", "ي"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            tok = tok[:-len(suf)]
            break
    return tok


_MARKS = re.compile(r"[\u064b-\u065f\u0670\u0640]")  # Arabic diacritics + tatweel


def _tokens(text: str) -> set[str]:
    # normalize first: sources write التضـخم with tatweel, claims without -
    # a glyph-level difference must not break entity/scope matching.
    text = _MARKS.sub("", text)
    raw = {_stem(t) for t in _WORD.findall(text.lower()) if t not in _STOP}
    return {_ALIASES.get(t, t) for t in raw}


def discover_sources(question: str, registry: list[Any],
                     generate: Callable[..., str] | None = None) -> dict[str, Any]:
    """Rank declared sources for a question. Registry items are SourceSpec-like
    objects with .name/.publisher and a .topics string (space/comma separated
    domain keywords supplied by the registry owner)."""
    q_tok = _tokens(question)
    scored = []
    for spec in registry:
        topics = getattr(spec, "topics", "") or ""
        t_tok = _tokens(f"{spec.name} {spec.publisher} {topics}")
        overlap = q_tok & t_tok
        scored.append({"name": spec.name, "publisher": spec.publisher,
                       "score": len(overlap), "matched": sorted(overlap), "spec": spec})
    scored.sort(key=lambda s: -s["score"])
    selected = [s for s in scored if s["score"] > 0]
    unselected = [{"name": s["name"], "publisher": s["publisher"],
                   "reason": "no topic/keyword overlap with the question"} for s in scored if s["score"] == 0]
    return {"selected": [s["spec"] for s in selected],
            "selected_detail": [{k: v for k, v in s.items() if k != "spec"} for s in selected],
            "unselected": unselected,
            "note": "declared registry only; no web crawling; model rerank not applied" if generate is None else "deterministic overlap; model rerank available"}
