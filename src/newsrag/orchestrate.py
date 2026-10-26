"""End-to-end investigation pipeline: question -> declared-source collection ->
facet extraction -> deterministic scope check -> sourced findings.

Every stage is library code; every figure in the output carries the provenance
record of the fetch it came from. What the pipeline cannot verify lands in
`gaps` with the reason - it is never smoothed over. This module assembles the
investigation STRUCTURE; long-form prose drafting is a separate stage.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from .collector import collect, SourceSpec
from .semantic import extract_facets, compare_facets

_FIGURE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|٪|مليون|مليار|ألف|آلاف|ملايين|حساب|حسابات|منشور|منشورات|تغريدة|تغريدات|دولار|جنيه|ريال|views?|impressions?|accounts?|posts?|percent)"
                     r"|\d{1,3}(?:,\d{3})+|\d{4,}")
# Qualitative evidence cues: coordination, attribution, denial, methodology.
# A sentence with no figure can still be the decisive evidence (e.g. a denial,
# a stated data source, a "no disclosure exists" scope statement).
_QUALITATIVE = re.compile(
    r"منسق|تنسيق|حملة|حسابات (?:وهمية|مزيفة)|وهمية|مزيفة|نفى|نفت|أنكر|أنكرت|"
    r"بحسب|وفقا|وفقاً|طبقا|طبقاً|منهجية|عينة|مصادر البيانات|أداة|"
    r"coordinat|inauthentic|campaign|methodolog|according to|denied|disclos", re.I)

# Metric scope words: what KIND of quantity a figure is. Reach/impressions are
# NOT unique people; views are NOT accounts; ad spend is NOT an operation budget.
_METRIC_KIND = [
    ("campaign_spend", re.compile(r"دولار|جنيه|ريال|درهم|spend|spent|advertis|\$", re.I)),
    ("campaign_accounts", re.compile(r"حساب|حسابات|accounts?", re.I)),
    ("campaign_volume", re.compile(r"منشور|منشورات|تغريدة|تغريدات|posts?|tweets?", re.I)),
    ("campaign_reach", re.compile(r"مشاهدة|مشاهدات|وصول|انطباع|انطباعات|reach|views?|impressions?", re.I)),
]


_CAMP_CONTEXT = re.compile(r"حملة|حملات|حساب|منشور|تغريد|campaign|account|post|tweet|advertis", re.I)


def metric_kind(sent: str) -> str | None:
    """Campaign-metric tag for evidence sentences. Requires campaign context:
    a GDP figure in US$ is not campaign spend."""
    if not _CAMP_CONTEXT.search(sent):
        return None
    for kind, pat in _METRIC_KIND:
        if pat.search(sent):
            return kind
    return None


def evidence_sentences(text: str, *, max_len: int = 600) -> list[dict[str, str]]:
    """Candidate evidence with its kind: quantitative (figure-bearing),
    qualitative (coordination/attribution/denial/methodology cues), or both.
    Qualitative sentences are first-class: a denial or a methodology statement
    is evidence even with no number in it."""
    out = []
    for sent in re.split(r"(?<=[.!؟?])\s+|\n+", text):
        sent = sent.strip()
        if not (20 <= len(sent) <= max_len):
            continue
        if ('{"' in sent or "widget_type" in sent or "data-" in sent
                or "fa-solid" in sent or sent.count("{") >= 2
                or sent.startswith("page:") or "per_page:" in sent or "sourceid:" in sent):
            continue  # HTML/JSON boilerplate or API envelope metadata, not content
        quant = bool(_FIGURE.search(sent))
        qual = bool(_QUALITATIVE.search(sent))
        if quant or qual:
            kind = "both" if quant and qual else ("quantitative" if quant else "qualitative")
            out.append({"text": sent, "kind": kind, "metric": metric_kind(sent) or ""})
    return out


_DECOMPOSE_PROMPT = """Break the RESEARCH QUESTION into 1-4 atomic, checkable CLAIMS that evidence could settle. Each claim must state one measure, one period, one scope - no compound questions. NEVER invent numbers, dates or figures: a claim may only contain values that appear in the question. Write the claims in the SAME LANGUAGE as the question. Return JSON only: {{"claims": ["...", ...]}}
Example QUESTION: ما معدل البطالة في مصر وما معدله في السعودية في 2025؟
Example JSON: {{"claims": ["ما معدل البطالة في مصر في 2025؟", "ما معدل البطالة في السعودية في 2025؟"]}}
RESEARCH QUESTION: {question}"""

_DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
    "required": ["claims"],
}




def _dedup_paraphrase_claims(claims: list[str]) -> list[str]:
    """A decomposed claim that is a token-subset paraphrase of a longer sibling
    adds no scope of its own (axis 2 = axis 1 minus the year, review):
    keep only the most specific claim of each paraphrase family."""
    from .discover import _tokens
    toks = [_tokens(c) for c in claims]
    return [c for i, c in enumerate(claims)
            if not toks[i]
            or not any(j != i and (toks[i] < toks[j] or (j < i and toks[i] == toks[j]))
                       for j in range(len(claims)))]

def decompose_question(question: str, generate_json: Callable[..., str] | None) -> list[str]:
    """Research question -> atomic evidence claims. A broad interrogative has no
    indicator of its own; comparing IT against evidence fabricates abstentions.
    Fail-closed: no usable decomposition means the caller reports the question as
    an explicit gap instead of checking a malformed pseudo-claim."""
    if generate_json is None:
        return []
    from .semantic import _extract_json, _excache_get, _excache_put, _decomp_cache_key, _DECOMP_STATS
    # decomposition at temperature 0 is deterministic and expensive (~78s on
    # small hardware): cached, keyed on model + prompt + grammar + question +
    # decomposition version - a change to any of them invalidates old entries
    # instead of replaying stale decompositions as fresh inference.
    _dk = _decomp_cache_key(question.strip()[:600])
    _dc = _excache_get(_dk)
    if _dc is not None:
        _DECOMP_STATS["hits"] += 1
        return _dedup_paraphrase_claims(list(_dc))
    _DECOMP_STATS["fresh"] += 1
    try:
        obj = _extract_json(str(generate_json(_DECOMPOSE_PROMPT.format(question=question.strip()[:600]))))
    except Exception:
        return []
    claims = obj.get("claims") if isinstance(obj, dict) else None
    if not isinstance(claims, list):
        return []
    q_numbers = set(re.findall(r"\d+(?:\.\d+)?", question))
    q_ar = len(re.findall(r"[\u0600-\u06FF]", question))
    q_lat = len(re.findall(r"[A-Za-z]", question))
    clean = []
    for c in claims:
        c = str(c).strip()[:400]
        if not c:
            continue
        # script guard: a claim in a different dominant script than the
        # question is model drift (observed: Arabic question -> Chinese
        # "claims"); it cannot scope-match real evidence.
        c_ar = len(re.findall(r"[\u0600-\u06FF]", c))
        c_lat = len(re.findall(r"[A-Za-z]", c))
        if q_ar > q_lat and c_ar <= c_lat:
            continue
        if q_lat > q_ar and c_lat <= c_ar:
            continue
        # a decomposed claim must not invent figures: any number it carries has
        # to come from the question itself, else the model hallucinated a value
        # and the "claim" would steer comparison toward a fabricated target.
        c_numbers = set(re.findall(r"\d+(?:\.\d+)?", c))
        if not c_numbers <= q_numbers:
            continue
        clean.append(c)
    _excache_put(_dk, clean[:4])
    return _dedup_paraphrase_claims(clean[:4])


_COMPARISON_SPLIT = re.compile(r"مقابل|مقارنة|versus|vs\.?|compared (?:to|with)", re.I)


_DATE_FRAG = re.compile(r"(?:^|[\s,(]){n}\s*,?\s*(?:19|20)\d\d|"
                        r"(?:January|February|March|April|May|June|July|August|September|"
                        r"October|November|December|يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|"
                        r"يوليو|أغسطس|اغسطس|سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر)\s+{n}\b")


def _drop_date_fragments(sentence: str, figures: list[str]) -> list[str]:
    """A day number in a date ('September 16, 2025') is not a measured figure;
    binding it as an answer fabricates a value. Drop figures sitting in date
    position in the source sentence."""
    keep = []
    for n in figures:
        if re.search(_DATE_FRAG.pattern.replace("{n}", re.escape(n)), sentence):
            continue
        keep.append(n)
    return keep


_DELTA_FIG = r"(?:percentage points?|نقطة مئوية|نقاط مئوية|درجة مئوية)"

_GRANULARITY = ("شهري", "سنوي", "ربع سنوي", "monthly", "annual", "quarterly")


def prefer_measure(ev_val, cl_val, cl_sent) -> str | None:
    """Evidence-side Arabic measure wins UNLESS it adds a granularity qualifier
    neither the claim measure nor the claim sentence carries: translation into
    Arabic invents those ("معدل البطالة الشهري" for a Q2 figure)."""
    ev, cl, sent = str(ev_val or ""), str(cl_val or ""), str(cl_sent or "")
    if ev and any('\u0600' <= ch <= '\u06FF' for ch in ev):
        for gran in _GRANULARITY:
            if gran in ev and gran not in cl and gran not in sent:
                return cl_val or None
        return ev_val
    return cl_val or ev_val or None


def dedup_answer_findings(findings: list[dict[str, Any]]) -> None:
    """Same claim + same source + same normalized bound value = one answer;
    later copies marked duplicate (reviewed case: headline + nav repetition both bound
    the figure; the card asserted it twice)."""
    seen: set[tuple] = set()
    for f in findings:
        if f.get("relation") != "answers" or not f.get("answer"):
            continue
        key = (f.get("claim"), f.get("source"),
               tuple(sorted(str(a).replace(",", "") for a in f["answer"])))
        if key in seen:
            f["relation"] = "duplicate"
            f["reason"] = "same claim/source/bound value as an earlier answers finding; not double-counted"
            f["reason_code"] = "duplicate_answer"
        else:
            seen.add(key)


def _drop_delta_figures(sentence: str, figures: list[str]) -> list[str]:
    """A change figure ("increasing by 0.1 percentage points") measures the
    movement, not the level: binding it alongside the rate asserts two
    answers (reviewed case: 6.5% level + 0.1pp delta both rendered as 'the rate')."""
    keep = []
    for n in figures:
        if (re.search(re.escape(n) + r"\s*" + _DELTA_FIG, sentence)
                or re.search(r"(?:by|بمقدار)\s*" + re.escape(n) + r"\s*" + _DELTA_FIG, sentence)
                or re.search(r"(?:by|بمقدار)\s*" + re.escape(n) + r"$", sentence.strip())):
            continue
        keep.append(n)
    return keep


_RAW_NUM = re.compile(r"[0-9][0-9,.]*")
_SCALE_WORD = re.compile(r"\s*(?:million|billion|thousand|mn|bn|mln|مليون|مليار|ملايين|ألف|آلاف)", re.IGNORECASE)


def _surface_norm(sentence: str, m) -> set[str]:
    """Normalized value of the figure at match m. The scale word belongs to
    the figure ONLY when it immediately follows it - a 20-char window bleeds
    into the NEXT figure ("2025, with 3.43 million" mapped 2025 -> 3430000,
"""
    from .semantic import _numbers
    surface = m.group(0).rstrip(".")
    sm = _SCALE_WORD.match(sentence[m.end():m.end() + 14])
    expanded = m.group(0) + (sm.group(0) if sm else "")
    norm = {x.rstrip(".") for x in _numbers([expanded])}
    norm.add(surface)
    return norm



def _drop_percent_for_nonrate(sentence: str, figures: list[str], claim_facets: dict) -> list[str]:
    """A percentage figure does not answer a count/level question: "بلغ 32.2
    مليون ... بمتوسط معدل نمو 2.5%" binds the population, not the 2.5%
    Only applies when the claim asks no rate/ratio."""
    from .semantic import _norm
    cm = _norm(claim_facets.get("measure"))
    if any(w in cm for w in _RATE_WORDS):
        return figures
    keep = []
    for n in figures:
        m = re.search(re.escape(str(n).rstrip(".")) + r"\s*[%٪]", sentence)
        if m:
            continue
        keep.append(n)
    return keep


def _drop_subset_figures(sentence: str, figures: list[str], claim_facets: dict) -> list[str]:
    """A figure explicitly scoped to a SUB-REGION ("with 3.43 million in the
    West Bank") does not answer a question about the whole ("State of
    Palestine"): binding it alongside the total asserts two answers.
    Drop a figure when its trailing context names a place outside the claim
    scope."""
    from .discover import _tokens
    scope = _tokens(str(claim_facets.get("population") or "") + " "
                    + str(claim_facets.get("entity") or "") + " "
                    + str(claim_facets.get("_sentence") or ""))
    from .semantic import _numbers
    wanted = {str(f).rstrip(".") for f in figures}
    keep_norm: set[str] = set()
    for m in _RAW_NUM.finditer(sentence):
        norm = _surface_norm(sentence, m)
        hit = norm & wanted
        if not hit:
            continue
        # trailing context of THIS occurrence: "in the West Bank" names a
        # sub-scope for the figure that precedes it
        tail = sentence[m.end():m.end() + 45]
        tm = re.match(r"\s*(?:million|billion|thousand|مليون|مليار|ألف)?\s*"
                      r"(?:in|of|في)\s+(?:the\s+)?([A-Za-z؀-ۿ][\w ]{2,25})", tail)
        if tm:
            place = _tokens(tm.group(1)) - {"population", "total", "state"}
            if place and not (place & scope):
                continue
        keep_norm |= hit
    return [f for f in figures if str(f).rstrip(".") in keep_norm]


def _document_anchor_years(reference_period: str) -> list[str]:
    """The document's substantive reference year(s): announcement/publication
    dates are metadata, stripped before counting (review). Inheritance is
    allowed only from a SINGLE remaining anchor year; a multi-period document
    anchors nothing, and a claim-asked year alone never creates an anchor."""
    refp = str(reference_period or "")
    stripped = re.sub(
        r"(?i)(?:announc\w*|releas\w*|publish\w*|issued)\s+(?:on\s+)?(?:results?\s+)?\w*\s*20\d\d(?:-\d\d-\d\d)?",
        lambda m: re.sub(r"20\d\d(?:-\d\d-\d\d)?", "", m.group(0)), refp)
    years = sorted(set(re.findall(r"20\d\d", stripped)))
    return years if len(years) == 1 else []


_SUBGROUP_LABEL = re.compile(
    r"غير\s+السعوديين|السعوديين|السعوديون|سعوديون|سعوديين|"
    r"non[- ]?Saudis?|Saudis?|"
    r"الضفة(?:\s+الغربية)?|West\s+Bank|قطاع\s+غزة|غزة|Gaza(?:\s+Strip)?|"
    r"ذكور|إناث|males?|females?|حضر|ريف|urban|rural|"
    r"مواطنون|مواطنين|citizens|مقيمون|مقيمين|residents|"
    r"الإجمالي|إجمالي|الإجمال|total", re.IGNORECASE)
_TOTAL_LABEL_TOK = {"إجمال", "total", "population", "عدد"}


def _drop_subgroup_mismatch(sentence: str, figures: list[str], claim_facets: dict) -> list[str]:
    """Per-figure scope anchor: the nearest subgroup label between the previous
    figure and this one (or in its short trailing context) names WHO the figure
    covers. "Saudis make up 18.8 million, while non-Saudis, 13.4 million" binds
    only 18.8M to a Saudis claim and nothing to a total-population claim;
    "32.2 Million, of Which 18.8 Million are Saudis" binds only 32.2M to the
    total (axis split). A figure with no local label is left as-is."""
    from .discover import _tokens
    pop = _tokens(str(claim_facets.get("population") or ""))
    sent_tok = _tokens(str(claim_facets.get("_sentence") or ""))
    totalish = (not pop or pop <= _TOTAL_LABEL_TOK
                or bool(sent_tok & {"إجمال", "total"}))
    wanted = {str(f).rstrip(".") for f in figures}
    keep_norm: set[str] = set()
    matches = list(_RAW_NUM.finditer(sentence))
    for i, m in enumerate(matches):
        norm = _surface_norm(sentence, m)
        hit = norm & wanted
        if not hit:
            continue
        head = sentence[matches[i - 1].end():m.start()] if i else sentence[:m.start()]
        labels = list(_SUBGROUP_LABEL.finditer(head))
        if not labels:
            tail = sentence[m.end():m.end() + 30]
            labels = list(_SUBGROUP_LABEL.finditer(tail))
        if not labels:
            keep_norm |= hit
            continue
        lab = labels[-1].group(0)
        neg = bool(re.match(r"(?i)non|غير", lab))
        base = _tokens(re.sub(r"(?i)^non[- ]?|^غير\s+", "", lab))
        if totalish:
            # only a total-scoped label (or none) answers the whole;
            # a subgroup-labelled figure never does
            if base and not (base <= _TOTAL_LABEL_TOK):
                continue
            keep_norm |= hit
            continue
        if neg and (base & pop):
            continue
        if not neg and base and not (base & pop):
            continue
        keep_norm |= hit
    return [f for f in figures if str(f).rstrip(".") in keep_norm]


def _year_local(sentence: str, figures: list[str], claim_facets: dict) -> list[str]:
    """A single sentence may carry figures for SEVERAL years ("grew from 24
    million in 2010 to 32.2 million in 2022"): when it names years other than
    the claim's, keep only figures stated near a claim-year mention - the rest
    belong to the other period and binding them fabricates answers (reviewed case:
    nine figures bound to one claim). Locality is measured on the figure's
    SURFACE form in the sentence (normalized values like 32200000 never
    appear literally where the source wrote 32.2 مليون)."""
    from .semantic import _numbers
    years = set(re.findall(r"(?:19|20)\d\d", str(claim_facets.get("time_period") or "")))
    sent_years = set(re.findall(r"(?:19|20)\d\d", sentence))
    if not years or not (sent_years - years):
        return figures
    wanted = {str(f).rstrip(".") for f in figures}
    keep_norm: set[str] = set()
    for m in _RAW_NUM.finditer(sentence):
        norm = _surface_norm(sentence, m)
        if not (norm & wanted):
            continue
        window = sentence[max(0, m.start() - 45):m.end() + 45]
        if any(y in window for y in years):
            keep_norm |= (norm & wanted)
    return [f for f in figures if str(f).rstrip(".") in keep_norm]


def _bind_answer(claim_facets: dict, sentence: str, evidence_facets: dict) -> list[str] | None:
    """Bind answer figures to the asked reference period. Returns the figure
    list for the matching segment, or None when binding is impossible."""
    from .semantic import _numbers, _period
    claim_period = _period(claim_facets.get("time_period"))
    segments = _COMPARISON_SPLIT.split(sentence)
    if len(segments) == 1:
        # single-segment evidence: its facet period already matched the claim
        # period in compare_facets, but the sentence itself may span years
        figs = _year_local(sentence, sorted(_numbers(evidence_facets.get("numbers"))), claim_facets)
        figs = _drop_percent_for_nonrate(sentence, figs, claim_facets)
        figs = _drop_subset_figures(sentence, figs, claim_facets)
        figs = _drop_subgroup_mismatch(sentence, figs, claim_facets)
        figs = _drop_delta_figures(sentence, _drop_date_fragments(sentence, figs))
        return _rescale_bound(sentence, figs) or None
    matched: list[str] = []
    for seg in segments:
        seg_period = _period(seg)
        if claim_period and seg_period and ({m for m, _ in claim_period} & {m for m, _ in seg_period}):
            matched.extend(sorted(_numbers([seg])))
    # a segment can still span two years ("ومقارنة بتعداد 2010م، فقد نما ... إلى
    # 32.2 مليون عام 2022م، فيما زاد عدد السعوديين من 14 إلى 18.8 مليون")
    matched = _year_local(sentence, matched, claim_facets)
    matched = _drop_percent_for_nonrate(sentence, matched, claim_facets)
    matched = _drop_subset_figures(sentence, matched, claim_facets)
    matched = _drop_subgroup_mismatch(sentence, matched, claim_facets)
    matched = _drop_delta_figures(sentence, _drop_date_fragments(sentence, matched))
    return _rescale_bound(sentence, matched) or None


def _rescale_bound(sentence: str, figures: list[str]) -> list[str]:
    """Bind figures at their sentence-stated scale: the extractor may list
    "35.3" while the sentence says "35.3 million" - the bound value must be
    the full 35300000, or downstream equality checks compare a mantissa
    against a population."""
    out = []
    for f in figures:
        fv = str(f).rstrip(".")
        best = f
        for m in _RAW_NUM.finditer(sentence):
            norm = _surface_norm(sentence, m)
            if fv in norm:
                expanded = [n for n in norm if n != m.group(0).rstrip(".")]
                if expanded:
                    try:
                        if abs(float(expanded[0])) >= abs(float(fv)):
                            best = expanded[0]
                    except ValueError:
                        pass
                break
        out.append(best)
    return out


_RATE_WORDS = ("معدل", "نسبة", "rate", "growth", "نمو")

_THRESHOLD = re.compile(
    r"exceeds?|exceeded|surpass(?:es|ed)?|over |more than|above |"
    r"تجاوز|تتجاوز|أكثر من|يزيد عن|فاق(?:ت)?", re.IGNORECASE)


def _entity_present(claim_facets: dict, sentence: str) -> bool:
    """A figure answer must come from a sentence that actually mentions the
    claim's entity: an inflation question is not answered by an index-level
    sentence with no 'inflation' in it. Bilingual stemmed tokens; skipped when
    the claim carries no usable entity."""
    from .discover import _tokens
    ent = _tokens(str(claim_facets.get("entity") or ""))
    generic = {"rate", "نسبة", "value", "figure", "prices"}
    ent -= generic
    if not ent:
        return True
    return bool(ent & _tokens(sentence))


def _answer_unit_filter(claim_facets: dict, sentence: str, bound: list[str]) -> list[str]:
    """A rate question cannot be answered by an index level: when the claim
    asks for a rate/ratio, bound figures must carry the % unit in the verbatim
    sentence, and a segment that only discusses the monthly variant does not
    answer the annual question."""
    from .semantic import _norm
    cm = _norm(claim_facets.get("measure"))
    if not any(w in cm for w in _RATE_WORDS):
        return bound
    kept = [n for n in bound if (f"{n}%" in sentence or f"{n} %" in sentence
                                 or f"{n}٪" in sentence or f"{n} percent" in sentence)]
    if ("سنوي" in cm or "annual" in cm) and ("شهري" in sentence or "monthly" in sentence)             and not ("سنوي" in sentence or "annual" in sentence):
        return []
    return kept


_RANK = {"supports": 3, "contradicts": 3, "answers": 3, "different_scope": 2, "insufficient": 1}


def _nums(nums):
    from .semantic import _numbers
    return _numbers(nums)


def dedup_paraphrase_claims(claims: list[str]) -> list[str]:
    """Decomposition sometimes emits the same axis reworded 3-4 times ("نهاية
    عام 2025" / "في عام 2025" / "نهاية السنة 2025"): each duplicate axis then
    renders its own false 'unanswered' block. Axes sharing >= 70% of
    their normalized tokens are the same question; keep the first."""
    from .discover import _tokens
    # Temporal/phrasing roots that distinguish only wording, not the asked
    # quantity ("نهاية عام 2025" vs "نهاية السنة 2025"). Anything else in the
    # token delta ("إجمال" vs none, "عدد" vs "نسبة") marks a DIFFERENT axis
    # that must survive (total-vs-Saudis; two-axis).
    safe_delta = {"سنة", "عام", "شهر", "يوم", "نهاية", "نهاي", "بداية",
                  "بداي", "منتصف", "تقدير", "تقريب"}
    kept: list[str] = []
    kept_tok: list[set] = []
    for c in claims:
        ct = _tokens(c)
        dup = any(
            len(ct & kt) / len(ct | kt) >= 0.7 and (ct ^ kt) <= safe_delta
            for kt in kept_tok
        ) if ct else False
        if dup:
            continue
        kept.append(c)
        kept_tok.append(ct)
    return kept



_COORD_CLAIM = re.compile(r"منسّ?ق|coordinat|orchestrat", re.I)
_COORD_ANALYSIS = re.compile(
    r"منسّ?ق[^.]{0,40}(حملة|شبكة|سلوك|تنسيق)|(حملة|شبكة|سلوك)[^ ]{0,3}[^.]{0,20}منسّ?ق|تنسيق مركزي|سلوك غير أصيل|حسابات وهمية|"
    r"centralized coordination|well[- ]coordinated|coordinated (campaign|network|behavior|behaviour|inauthentic|posts?|accounts?)|"
    r"inauthentic behavior|orchestrated (campaign|network)|coordination (network|patterns?|model)|signs of coordination",
    re.I)
_CAPTION_CUE = re.compile(r"^\s*صورة|by reviewing the (image|photo)|photo (circulating|shared)|image circulating|"
                          r"^\d{2}, \d{4}\b", re.I)


def _coordination_required(claim_text: str) -> bool:
    """The claim asks about COORDINATED activity specifically (منسقة/coordinated).
    Recorded from the claim TEXT, never inferred by the model."""
    return bool(_COORD_CLAIM.search(claim_text or ""))


def _is_caption_or_heading(sent: str) -> bool:
    """Photo captions, headings and article-card fragments are not analysis:
    a caption that repeats a campaign slogan is not evidence OF coordination."""
    t = (sent or "").strip()
    if _CAPTION_CUE.search(t):
        return True
    return len(t.split()) < 7


_COORD_WORD = re.compile(r"منسّ?ق|coordinat|orchestrat", re.IGNORECASE)
_HEADING_VERB = re.compile(
    r"\b(indicates?|reveals?|shows?|found|concludes?|demonstrates?|suggests?"
    r"|يكشف|تُظهر|توضح|تبيّن|تبين|خلص|خلصت|استنتج|استنتجت|يرصد|يوثّق|يوثق)\b",
    re.IGNORECASE)


def _coordination_grade(sent: str) -> bool:
    """Analysis-level coordination content: the sentence itself analyzes
    coordination/network behavior (not a caption, heading, methodology note,
    or an inference about shared sentiment/regional alignment)."""
    if _is_caption_or_heading(sent):
        return False
    if not _COORD_ANALYSIS.search(sent or ""):
        return False
    # heading + methodology is not an analysis conclusion: when the
    # coordination cue sits in a leading verb-less title segment ("Indicators
    # of a Coordinated Campaign Using Meltwater, the data collection..."),
    # the sentence reports process, not a coordination finding (review).
    first = re.split(r"[,،—]", sent or "", maxsplit=1)[0]
    if _COORD_WORD.search(first) and not _HEADING_VERB.search(first):
        return False
    return True

def run_investigation(question: str, registry: list[SourceSpec], generate: Callable[..., str],
                      *, decompose_generate: Callable[..., str] | None = None,
                      opener: Callable | None = None, max_evidence_per_source: int = 8,
                      recorder: list | None = None,
                      discovery_trace: list[str] | None = None) -> dict[str, Any]:
    """Run the pipeline. `generate` is a bound local-model generator (see
    newsrag.qwen.llamacpp_generate); nothing leaves the machine. When
    `recorder` (a list) is passed, timestamped stage events are appended as
    {t, text} dicts - the raw cast an honest trial video is rendered from."""
    import time as _time
    _t0 = _time.monotonic()
    def rec(text: str) -> None:
        if recorder is not None:
            recorder.append({"t": round(_time.monotonic() - _t0, 2), "text": text})
    rec(f"QUESTION: {question}")
    for line in (discovery_trace or []):
        rec(f"discovery: {line}")
    rec(f"sources: {[s2.name for s2 in registry]}")
    bundle = collect(question, registry, opener=opener)
    for pr in bundle["provenance"]:
        rec(f"collect {pr['status']}: {pr['source']} ({pr['detail'][:80]})")
    claims = dedup_paraphrase_claims(decompose_question(question, decompose_generate))
    question_frame = False
    for cl in claims:
        rec(f"claim: {cl[:100]}")
    if not claims:
        # fallback for a single-fact question: the information need itself, used
        # as a scope template (never as a truth claim). It has no figures or
        # polarity; evidence that matches its scope and supplies the figure
        # ANSWERS it (relation "answers"), distinct from fact-check "supports".
        claims = [question.strip()]
        question_frame = True
    findings: list[dict[str, Any]] = []
    gaps: list[dict[str, str]] = []
    claim_facets = []
    for c in claims:
        cf = extract_facets(c, generate)
        if cf is not None:
            # sanitize extractor noise: a bare "null"/"none" string is not a
            # figure, and a question carries no direction - a polarity the model
            # invents for an interrogative would veto genuine answers.
            nums = cf.get("numbers")
            if isinstance(nums, list):
                cf["numbers"] = [n for n in nums
                                 if isinstance(n, str) and n.strip().lower() not in
                                 ("null", "none", "n/a", "") and re.search(r"\d", n)]
            if isinstance(cf.get("currency"), str) and cf["currency"].strip().lower() in ("null", "none", "n/a", ""):
                cf["currency"] = None  # literal "null" string is extractor noise, not a currency
            # a decomposed claim cannot invent a year absent from the question
            # (reviewed case: "الأشهر السبعة الأولى من 2026" became "أواخر 2025")
            q_years = set(re.findall(r"20\d\d", question))
            c_years = set(re.findall(r"20\d\d", str(cf.get("time_period") or "")))
            if c_years and q_years and not (c_years & q_years):
                # fall back to the question's own year(s), not the invention
                cf["time_period"] = "/".join(sorted(q_years))
            elif not c_years and q_years and c.strip().endswith("؟"):
                # a decomposed sub-question inherits the parent question's year
                # ("هل وثّقت ... خلال عام 2025؟" decomposes into sub-questions
                # that drop the year; period_missing regression)
                cf["time_period"] = (str(cf.get("time_period") or "") + " "
                                     + " ".join(sorted(q_years))).strip()
                cf["_period_inherited"] = "question"
            # granularity qualifier the question never stated is an invention
            # ("معدل البطالة الشهري" for a quarterly question)
            q = question
            for gran in _GRANULARITY:
                if cf.get("measure") and gran in str(cf["measure"]) and gran not in q and gran not in c:
                    cf["measure"] = re.sub(r"\s*" + gran, "", str(cf["measure"])).strip()
            cf["_sentence"] = c  # scope fallback may use the claim text
            # Decomposed interrogatives ("ما معدل ...؟") are question frames too:
            # they carry no direction or metric kind, so a model-invented
            # polarity would veto genuine answers as fake contradictions.
            if question_frame or c.strip().endswith("؟"):
                cf["polarity"] = None
                # a figure-less question has no metric kind; a model-invented
                # one (population tagged "accounts") vetoes genuine answers.
                cf["metric_kind"] = None
        claim_facets.append({"claim": c, "facets": cf,
                             "question_frame": question_frame or c.strip().endswith("؟")})
        if cf is None:
            gaps.append({"stage": "claim_analysis", "claim": c,
                         "reason": "could not extract scope facets from this claim"})
    unrelated = 0
    for item in bundle["gathered"]:
        cands = evidence_sentences(item["text"])
        if claim_facets and len(cands) > max_evidence_per_source:
            # retrieval-first ranking, PER CLAIM: each axis of a multi-axis
            # question gets its own top-k before the (slow) facet extractor, so
            # one axis's evidence cannot crowd out another's. A combined-token
            # ranking let a shared monthly-headline sentence consume the whole
            # budget and starve the annual all-republic axis (defect).
            from .discover import _tokens
            picked: list[dict[str, str]] = []
            seen: set[int] = set()
            for c in claim_facets:
                ctok = _tokens(c["claim"])
                # qualitative question: the source's DECLARED subject keywords
                # are legitimate extra ranking signal for qualitative-kind
                # candidates - analytic sentences ("9,000 posts on April 5",
                # "57% reposts", "signs of inauthentic behavior") share almost
                # no tokens with an Arabic claim and were starved out of the
                # per-claim budget below headline/lead lines (observed failure:
                # 10 of 11 coordination-indicator candidates ranked 18-69).
                ttok = (_tokens(str(item.get("topics") or ""))
                        if c.get("question_frame") else set())
                ranked = sorted(
                    cands,
                    key=lambda c0: -(len(_tokens(c0["text"]) & ctok)
                                     + (len(_tokens(c0["text"]) & ttok)
                                        if c0["kind"] in ("qualitative", "both") else 0)))
                for c0 in ranked[:max_evidence_per_source]:
                    if id(c0) not in seen:
                        seen.add(id(c0))
                        picked.append(c0)
            # qualitative questions: token ranking starves the analytic /
            # indicator sentences that carry the actual findings (observed: 10 of
            # 11 coordination-indicator candidates ranked 18-69 against Arabic
            # claim tokens). When the question is a qualitative frame AND the
            # source's candidate pool is genuinely mixed (>= 35% qualitative),
            # widen coverage: per-claim top-k, then a qualitative slice, a
            # quantitative slice, and a fill pass by claim+topic overlap up to
            # a hard per-source cap. Numeric pages (qualitative share low)
            # keep the strict token-ranked budget. Every widened candidate
            # still passes the same facet comparison and abstention guards.
            if (any(c.get("question_frame") for c in claim_facets)
                    and cands
                    and sum(1 for c0 in cands if c0["kind"] in ("qualitative", "both"))
                    >= 0.35 * len(cands)):
                ttok_s = _tokens(str(item.get("topics") or ""))
                qtok_s: set = set()
                for c in claim_facets:
                    qtok_s |= _tokens(c["claim"])
                cap = min(len(cands), max(48, 4 * max_evidence_per_source))
                qual_rest = [c0 for c0 in cands
                             if id(c0) not in seen and c0["kind"] == "qualitative"]
                qual_rest.sort(key=lambda c0: -len(_tokens(c0["text"]) & (qtok_s | ttok_s)))
                for c0 in qual_rest[:max_evidence_per_source]:
                    seen.add(id(c0))
                    picked.append(c0)
                quant_rest = [c0 for c0 in cands
                              if id(c0) not in seen
                              and c0["kind"] in ("quantitative", "both")]
                quant_rest.sort(key=lambda c0: -len(_tokens(c0["text"]) & (qtok_s | ttok_s)))
                for c0 in quant_rest[:max_evidence_per_source]:
                    seen.add(id(c0))
                    picked.append(c0)
                # fill pass: still scope-bound - a sentence sharing NO token
                # with the claims or the source's declared topics is out of
                # scope, not merely low-ranked (also caps extraction cost).
                rest = [c0 for c0 in cands
                        if id(c0) not in seen and (_tokens(c0["text"]) & (qtok_s | ttok_s))]
                rest.sort(key=lambda c0: -len(_tokens(c0["text"]) & (qtok_s | ttok_s)))
                for c0 in rest:
                    if len(picked) >= cap:
                        break
                    seen.add(id(c0))
                    picked.append(c0)
            cands = picked
        else:
            cands = cands[:max_evidence_per_source]
        related = 0
        for cand in cands:
            sent = cand["text"]
            ef = extract_facets(sent, generate) if claim_facets else None
            if ef is not None:
                ef["_sentence"] = sent  # scope fallback may use the raw text
            if ef is not None and item.get("reference_period"):
                refp = str(item.get("reference_period") or "")
                if not ef.get("time_period"):
                    # a figure inside a dated release refers to that release date
                    # unless the sentence says otherwise (MPC held-rate lines carry
                    # no date of their own)
                    ef["time_period"] = refp
                    ef["_period_inherited"] = "document"
                    ef["_period_antecedent"] = refp
                elif not re.findall(r"20\d\d", str(ef.get("time_period") or "")):
                    # a year-less extractor period ("العام") is as uninformative as
                    # a missing one: scope the figure to the release's year(s),
                    # preferring years a claim actually asks about (reviewed case: "Census
                    # 2022 (announced 2023-05-31)" -> 2022 for the 2022 claims).
                    # announcement/publication dates are document metadata, not
                    # the reference period: strip years stated as the announced/
                    # released/published date before counting document years
                    # (a claim-asked year alone is not a document
                    # evidence year).
                    # explicit antecedent only (independent review): the
                    # DOCUMENT must anchor the year - a single substantive
                    # reference year. A claim asking about a year a multi-period
                    # document merely mentions is NOT an anchor; do not inherit.
                    use = _document_anchor_years(refp)
                    if use:
                        ef["time_period"] = " ".join(use)
                        ef["_period_inherited"] = "document-year"
                        ef["_period_antecedent"] = refp
            if ef is None:
                findings.append({"source": item["source"], "url": item["url"],
                                 "evidence": sent, "evidence_kind": cand["kind"],
                                 "metric": cand["metric"], "relation": "insufficient",
                                 "reason": "evidence extraction failed; recorded, not counted"})
                continue
            best = None
            best_c = None
            for c in claim_facets:
                if c["facets"] is None:
                    continue
                v = compare_facets(c["facets"], ef)
                from .semantic import _SCOPE_PATH
                v["scope_path"] = _SCOPE_PATH[0]
                if (c.get("question_frame") and v.get("reason_code") == "figures_missing"
                        and (_nums(ef.get("numbers")) or cand["kind"] in ("qualitative", "both"))):
                    # question-frame matched scope/period; evidence supplies the figure.
                    # But an evidence sentence may carry SEVERAL figures for SEVERAL
                    # periods ("12.8% in September vs 12.7% in August"): bind the
                    # answer to the segment whose period matches the claim period,
                    # else a comparison-month figure leaks in as "the answer".
                    pre_nums = sorted(_nums(ef.get("numbers")))
                    if _THRESHOLD.search(sent):
                        # threshold statement ("Total population EXCEEDS 35
                        # million"): the figure is a bound, not the measured
                        # value - preserving it as an asserted point estimate
                        # would over-bind (reviewed case). It remains context.
                        bound = None
                    elif ef.get("_origin") in ("record-line-parser", "en-stat-parser"):
                        # A collector-rendered record line binds label+date+value
                        # by construction, and compare_facets already matched its
                        # indicator/scope/period; prose guards (period segment,
                        # unit glyph, entity wording) do not apply to fields.
                        bound = sorted(_nums(ef.get("numbers"))) or None
                    else:
                        bound = _bind_answer(c["facets"], sent, ef)
                        if bound is not None:
                            bound = _answer_unit_filter(c["facets"], sent, bound) or None
                        if bound is not None and not _entity_present(c["facets"], sent):
                            bound = None
                    if bound is None and cand["kind"] in ("qualitative", "both") and _coordination_required(c["claim"]) and not _coordination_grade(sent):
                        # coordination claim, caption/inference-level evidence:
                        # mention recorded, NOT an answer (review)
                        v = {"relation": "insufficient", "dimension": None, "reason_code": "coordination_bar",
                             "answer": None,
                             "reason": "coordination claim requires analysis-level coordination evidence; caption/heading/inference-level mention recorded, not counted"}
                    elif bound is None and cand["kind"] in ("qualitative", "both"):
                        v = {"relation": "answers", "dimension": None, "reason_code": "answers_qualitative",
                             "answer": None,
                             "reason": ("qualitative evidence matches the question's scope/period"
                                        + ("; figures present but not bindable to the asked period" if _nums(ef.get("numbers")) else "")
                                        + ("; coordination-grade analysis" if _coordination_required(c["claim"]) else ""))}
                    elif bound is None and _THRESHOLD.search(sent) and _nums(ef.get("numbers")):
                        v = {"relation": "answers", "dimension": None, "reason_code": "answers_qualitative",
                             "answer": None,
                             "reason": "threshold statement (exceeds/over) preserved as context; its bound figure is not the measured value"}
                    elif bound is None and _nums(ef.get("numbers")):
                        v = {"relation": "insufficient", "dimension": None, "reason_code": "answer_unbound",
                             "reason": "evidence figures could not be bound to the asked period; abstaining"}
                    elif bound is None and _coordination_required(c["claim"]) and not _coordination_grade(sent):
                        v = {"relation": "insufficient", "dimension": None, "reason_code": "coordination_bar",
                             "answer": None,
                             "reason": "coordination claim requires analysis-level coordination evidence; caption/heading/inference-level mention recorded, not counted"}
                    elif bound is None:
                        v = {"relation": "answers", "dimension": None, "reason_code": "answers_qualitative",
                             "answer": None,
                             "reason": ("qualitative evidence matches the question's scope/period (no figure to bind)"
                                        + ("; coordination-grade analysis" if _coordination_required(c["claim"]) else ""))}
                    else:
                        lex = _SCOPE_PATH[0] != "indicator"
                        v = {"relation": "answers", "dimension": None, "reason_code": "answers",
                             "answer": bound, "scope_path": _SCOPE_PATH[0],
                             "reason": ("evidence matches the question's scope/period via LEXICAL overlap "
                                        "(provisional, scope not verified) and supplies the figure" if lex
                                        else "evidence matches the question's verified scope/period and supplies the figure")}
                if best is None or _RANK[v["relation"]] > _RANK[best["relation"]]:
                    best = {"claim": c["claim"], **v}
                    best_c = c
            if best is None:
                continue
            if best["relation"] in ("supports", "contradicts", "answers", "different_scope"):
                related += 1
            else:
                unrelated += 1
            frec = {"source": item["source"], "url": item["url"],
                    "evidence": sent, "evidence_kind": cand["kind"],
                    "metric": cand["metric"], **best}
            rec(f"finding [{best['relation']}] {sent[:80]}")
            if best["relation"] == "answers":
                narrowed = bool(best.get("answer")) and \
                    {str(x).rstrip(".") for x in best["answer"]} != \
                    {str(x).rstrip(".") for x in pre_nums} if best.get("answer") is not None else False
                # the rendered answer must carry unit/measure/period/scope, not
                # a bare number - sourced QA without them is not an answer.
                def _ar(ev_val, cl_val):
                    # rendered scope/measure/period must be readable Arabic:
                    # prefer the evidence facet, fall back to the (Arabic) claim
                    # facet when the evidence facet is a latin extraction fragment
                    ev = str(ev_val or "")
                    if ev and any('\u0600' <= ch <= '\u06FF' for ch in ev):
                        return ev_val
                    return cl_val or ev_val or None
                cf_win = (best_c or {}).get("facets") or {}
                ev_tp, cl_tp = str(ef.get("time_period") or ""), str(cf_win.get("time_period") or "")
                tp = ev_tp if len(ev_tp) > len(cl_tp) else (cl_tp or ev_tp)  # granularity wins
                if narrowed:
                    # binding narrowed to the claim's local scope (year-local /
                    # subset / percent guards): the figure answers the CLAIM's
                    # period and population, not the sentence's wider span
                    # (reviewed case: 32.2M is 2022, not "2010 إلى 2022"; reviewed case: 5.56M is
                    # the State of Palestine, not "السكان في West Bank").
                    tp = cl_tp or ev_tp
                pop_val = ((cf_win.get("population") or ef.get("population"))
                           if narrowed else _ar(ef.get("population"), cf_win.get("population")))
                frec["answer_context"] = {"measure": prefer_measure(ef.get("measure"), cf_win.get("measure"), cf_win.get("_sentence")),
                                          "time_period": tp or None,
                                          "population": pop_val,
                                          "entity": _ar(ef.get("entity"), cf_win.get("entity")),
                                          "currency": ef.get("currency"),
                                          "unit": ef.get("unit")}
                # per-finding honesty record (independent review): the period's basis
                # (verbatim in the sentence vs inherited from the document's
                # declared reference period) and WHICH raw surface each bound
                # value matched. A document-inherited period is not verbatim;
                # the check harness must not claim it is.
                frec["period_basis"] = ("document-inherited"
                                        if ef.get("_period_inherited") else "verbatim")
                if ef.get("_period_antecedent"):
                    frec["period_antecedent"] = ef["_period_antecedent"]
                if best.get("answer"):
                    ms = []
                    for bv in best["answer"]:
                        bvn = str(bv).rstrip(".")
                        surf = None
                        for m in _RAW_NUM.finditer(sent):
                            if bvn in _surface_norm(sent, m):
                                _sm = _SCALE_WORD.match(sent[m.end():m.end() + 14])
                                surf = (m.group(0) + (_sm.group(0) if _sm else "")).strip()
                                break
                        ms.append({"value": bv,
                                   "match": ("stated-surface" if surf else
                                             ("parser-field" if ef.get("_origin") in
                                              ("record-line-parser", "en-stat-parser")
                                              else "normalized-only")),
                                   **({"surface_text": surf} if surf else {})})
                    frec["matched_surface"] = ms
            findings.append(frec)
        if cands and related == 0:
            gaps.append({"stage": "relation", "source": item["source"],
                         "reason": f"{len(cands)} evidence sentences found but none could be related "
                                   f"to the decomposed claims (scope taxonomy or extraction limit)"})
        if not cands:
            gaps.append({"stage": "evidence_scan", "source": item["source"],
                         "reason": "no evidence-bearing sentences found in fetched text"})
    for pr2 in bundle["provenance"]:
        if pr2["status"] != "ok":
            gaps.append({"stage": "collection", "source": pr2["source"],
                         "reason": f"{pr2['status']}: {pr2['detail']}"})
    dedup_answer_findings(findings)
    n_ans = sum(1 for f in findings if f.get("relation") == "answers")
    rec(f"findings={len(findings)} answers={n_ans} gaps={len(gaps)}")
    return {"question": question,
            "decomposed_claims": claim_facets,
            "findings": findings,
            "gaps": gaps,
            "provenance": bundle["provenance"]}
