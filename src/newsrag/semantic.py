"""Semantic claim-evidence relation check: meaning over token overlap.

Classifies (claim, support) as supports / contradicts / different_scope /
insufficient, naming the mismatch dimension (entity, measure, time,
population, attribution, direction, meaning) when the scope differs.
Runs on the same local model as the rest of the library; every verdict is
an auditable trace record, never silently merged with lexical support.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

RELATIONS = ("supports", "contradicts", "different_scope", "insufficient")
DIMENSIONS = ("entity", "measure", "time", "population", "attribution", "direction", "meaning")

_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": list(RELATIONS)},
        "dimension": {"type": ["string", "null"], "enum": list(DIMENSIONS) + [None]},
        "reason": {"type": "string"},
    },
    "required": ["relation", "dimension", "reason"],
}

_PROMPT = """You are a precise evidence examiner. Decide whether the EVIDENCE supports the CLAIM, contradicts it, talks about a different scope, or is insufficient.

Rules:
- supports: the evidence affirms the claim's meaning, even with different words or another language (e.g. تراجع = decelerated).
- contradicts: the evidence affirms the opposite meaning. Direction matters: if the rate FELL, a claim that it ROSE contradicts. "Inflation fell" does NOT mean "prices fell".
- different_scope: same topic and maybe the same number, but a different entity, measure (e.g. all-Republic vs urban vs core), time period, population, or attribution (who said it).
- insufficient: the evidence does not settle the claim at all.

Match entity, measure, time period, population and attribution before accepting a matching number. Numbers alone prove nothing.

Return JSON only: {{"relation": "...", "dimension": "...", "reason": "one short sentence"}}
dimension: one of entity, measure, time, population, attribution, direction, meaning - or null when relation is supports or insufficient.

CLAIM: {claim}
EVIDENCE: {support}"""


def _extract_json(text: str) -> dict[str, Any] | None:
    """Last decodable JSON object wins: llama-cli echo truncation can leave
    prompt fragments (including few-shot example JSON) before the reply, so a
    greedy first-brace match can glue echo and reply into unparseable text or
    return the example instead of the answer."""
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\{", text))):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


SCHEMA = _SCHEMA


def verify_relation(claim: str, support: str, generate: Callable[..., str]) -> dict[str, Any]:
    """Classify the claim-evidence relation with the local model.

    Returns {relation, dimension, reason, valid}: valid=False when the model
    output was unusable; callers must treat that as unknown, never as support.
    """
    prompt = _PROMPT.format(claim=claim.strip()[:600], support=support.strip()[:900])
    raw = str(generate(prompt))  # caller binds constrained decoding via json_schema=_SCHEMA
    obj = _extract_json(raw) or {}
    rel = obj.get("relation")
    dim = obj.get("dimension")
    reason = str(obj.get("reason") or "").strip()[:300]
    if rel not in RELATIONS:
        return {"relation": "insufficient", "dimension": None,
                "reason": "model verdict unusable; treated as unknown", "valid": False}
    if rel in ("supports", "insufficient"):
        dim = None
    elif dim not in DIMENSIONS:
        dim = None
    return {"relation": rel, "dimension": dim, "reason": reason, "valid": True}


# --- Facet extraction + deterministic comparison -------------------------
# The local model extracts scope facets (a task it does reliably); plain
# code compares them (a task models do unreliably). Verdicts stay auditable.

FACET_SCHEMA = {
    "type": "object",
    "properties": {
        "entity": {"type": ["string", "null"]},
        "measure": {"type": ["string", "null"]},
        "time_period": {"type": ["string", "null"]},
        "population": {"type": ["string", "null"]},
        "attribution": {"type": ["string", "null"]},
        "polarity": {"type": ["string", "null"], "enum": ["up", "down", "flat", "none", None]},
        "numbers": {"type": "array", "items": {"type": "string"}},
        "metric_kind": {"type": ["string", "null"],
                        "enum": ["spend", "accounts", "volume", "reach", "other", "none", None]},
        "currency": {"type": ["string", "null"]},
        "figure_source": {"type": ["string", "null"],
                          "enum": ["platform_disclosure", "researcher_estimate",
                                   "official_statement", "unknown", None]},
    },
    "required": ["entity", "measure", "time_period", "population", "attribution", "polarity",
                 "numbers", "metric_kind", "currency", "figure_source"],
}

_FACET_PROMPT = """Extract scope facets from the TEXT as JSON only.
entity: what the figure is about. measure: the EXACT indicator, preserving qualifiers like monthly/annual, urban/all-republic, core/headline - never generalize a specific indicator into a broader one. time_period: the reference month/period WITH year if stated. population: who/where it covers, exactly as stated. attribution: who published/said it, if stated. polarity: does the TEXT say the indicator went up, down, flat, or none (none when only a level is stated). numbers: every percentage/figure mentioned, as strings.
metric_kind: the KIND of quantity - spend (money paid), accounts (number of accounts), volume (number of posts/tweets/items), reach (views/impressions/reach - potential exposure, NEVER unique people), other, or none. currency: ISO code (USD, SAR, EGP, AED...) when money is stated, else null. figure_source: who produced the figure - platform_disclosure (the platform's own transparency/takedown report), researcher_estimate (a researcher's/tool's analysis, e.g. Meltwater), official_statement, or unknown. Reach is not people; ad spend is not an operation's total budget.
Example TEXT: ارتفع معدل البطالة الشهري في المدن إلى 7.1% في مارس 2026.
Example JSON: {{"entity": "البطالة", "measure": "معدل البطالة الشهري", "time_period": "مارس 2026", "population": "المدن", "attribution": null, "polarity": "up", "numbers": ["7.1%"], "metric_kind": "other", "currency": null, "figure_source": "unknown"}}
TEXT: {text}"""

_POP_URBAN = ("urban", "الحضر", "المدن", "cities", "city")
_POP_ALLREP = ("all republic", "all-republic", "إجمالي الجمهورية", "الجمهورية", "national", "whole country")
_MEAS_CORE = ("core", "الأساسي")
_MEAS_MONTHLY = ("monthly", "الشهري", "شهرى", "month-on-month", "month over month")
_MEAS_HEADLINE = ("headline", "العام السنوي", "السنوي", "annual")
_ATTR_CBE = ("central bank", "البنك المركزي", "bank of egypt", "cbe")
_ATTR_CAPMAS = ("capmas", "الجهاز المركزي للتعبئة", "للتعبئة العامة والإحصاء", "الإحصاء")

_MONTHS = {"يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "ابريل": 4, "مايو": 5, "يونيو": 6,
           "يوليو": 7, "أغسطس": 8, "اغسطس": 8, "سبتمبر": 9, "أكتوبر": 10, "اكتوبر": 10,
           "نوفمبر": 11, "ديسمبر": 12,
           "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
           "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
           "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
           "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}


_ALEF = str.maketrans({"\u0623": "\u0627", "\u0625": "\u0627", "\u0622": "\u0627",
                       "\u0649": "\u064a"})


def _norm(s: Any) -> str:
    # alef/hamza and alef-maqsura variants are spelling, not meaning: the 1.5B
    # writes الامية where the question writes أمية (seen miss)
    t = str(s or "").translate(_ALEF)
    return re.sub(r"\s+", " ", t.strip().lower()).strip(" .,%")


def _canon(s: Any, groups: tuple[tuple[str, ...], ...]) -> str | None:
    """Map a free string to its alias-group index, else the normalized string."""
    n = _norm(s)
    if not n or n in ("not specified", "not explicitly stated", "none", "unknown", "غير مذكور", "غير محدد", "غير مذكورة", "غير محددة"):
        return None
    for i, g in enumerate(groups):
        if any(a in n for a in g):
            return f"g{i}"
    return n


_MULT = (("مليار", 1e9), ("billion", 1e9), ("ملايين", 1e6), ("مليون", 1e6),
         ("million", 1e6), ("آلاف", 1e3), ("ألف", 1e3), ("thousand", 1e3), ("k", 1e3))


def _canon_num(value: float) -> str:
    return str(int(value)) if value == int(value) else ("%.6f" % value).rstrip("0")


def _numbers(nums: Any) -> set[str]:
    """Figures normalized with multiplier words: '1.4 million' == '1.4 مليون' ==
    '1400000'. Bare 'k' only counts right after a digit."""
    out = set()
    for x in nums or []:
        text = str(x)
        for m in re.finditer(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s*([A-Za-z\u0600-\u06FF]+))?", text):
            tok = m.group(1).replace(",", "")  # 57,348.2 is one figure, not two
            if re.fullmatch(r"(19|20)\d\d", tok):
                continue  # a bare year is a date reference, not a figure
            word = (m.group(2) or "").lower()
            mult_applied = False
            val = float(tok)
            for w, mult in _MULT:
                if word == w or (w in ("k",) and word == "k"):
                    val *= mult
                    mult_applied = True
                    break
            if mult_applied:
                out.add(_canon_num(val))
            else:
                # no multiplier: keep the original token - float() round-trips
                # lose precision on large values (389059910593.1 -> ...09976).
                out.add(tok.rstrip("0").rstrip(".") if "." in tok else tok)
    return out


def _period(s: Any) -> set[tuple[int, ...]]:
    n = _norm(s)
    months = {m for name, m in _MONTHS.items()
              if re.search(r"(?<![A-Za-z؀-ۿ])" + re.escape(name) + r"(?![A-Za-z؀-ۿ])", n)}
    years = {int(y) for y in re.findall(r"20\d\d", n)}
    # ISO dates carry their own month: "Decision of 2026-09-24"
    for iso_y, iso_m in re.findall(r"(20\d\d)-(\d\d)-\d\d", n):
        months.add(int(iso_m))
        years.add(int(iso_y))
    return {(mo, yr) for mo in months for yr in (years or {0})} or ({(0, yr) for yr in years} if years else set())


_DEGENERATE = {"percentage", "rate", "value", "figure", "number",
               "النسبة", "النسبة المئوية", "المعدل", "القيمة", "الرقم"}


_EXCACHE: dict[str, Any] | None = None
_EXCACHE_PATH: str | None = None


def _excache_get(key: str):
    global _EXCACHE, _EXCACHE_PATH
    import os, json as _json, hashlib as _h
    if _EXCACHE is None:
        _EXCACHE_PATH = os.environ.get("NEWSRAG_EXCACHE")
        _EXCACHE = {}
        if _EXCACHE_PATH and os.path.exists(_EXCACHE_PATH):
            try:
                _EXCACHE = _json.load(open(_EXCACHE_PATH))
            except Exception:
                _EXCACHE = {}
    return _EXCACHE.get(key)


def _excache_put(key: str, value: Any) -> None:
    import json as _json
    _excache_get(key)  # ensure loaded
    _EXCACHE[key] = value
    if _EXCACHE_PATH:
        try:
            _json.dump(_EXCACHE, open(_EXCACHE_PATH, "w"), ensure_ascii=False)
        except Exception:
            pass


_DECOMP_VERSION = "decomp-v2"  # bump when claims guards/shape change
_DECOMP_STATS: dict[str, int] = {"hits": 0, "fresh": 0}


def _decomp_cache_key(question: str) -> str:
    """Decomposition cache key: binds model, decompose prompt, claims grammar
    and the decomposition-guard version in addition to the question text, so
    a change to any of them invalidates stale entries. Distinct namespace from
    facet-extraction keys; _DECOMP_STATS records hit/fresh counts for cost
    accounting."""
    import hashlib as _hh, os
    from .orchestrate import _DECOMPOSE_PROMPT
    h = _hh.md5()
    h.update(question.encode())
    h.update(_DECOMPOSE_PROMPT.encode())
    gf = os.environ.get("NEWSRAG_CLAIMS_GRAMMAR", "examples/adversarial/claims.gbnf")
    try:
        h.update(open(gf, "rb").read())
    except OSError:
        h.update(b"no-grammar")
    h.update(os.environ.get("NEWSRAG_MODEL_ID", "qwen15.gguf").encode())
    h.update(_DECOMP_VERSION.encode())
    return "decomp:" + h.hexdigest()


def _cache_key(text: str) -> str:
    """Cache key binds model, prompt, grammar, facet schema AND input text:
    a fix to any of them invalidates old entries instead of replaying stale
    extractions as fresh inference."""
    import hashlib as _hh, os
    h = _hh.md5()
    h.update(text.strip()[:700].encode())
    h.update(_FACET_PROMPT.encode())
    gf = os.environ.get("NEWSRAG_FACET_GRAMMAR", "examples/adversarial/facet.gbnf")
    try:
        h.update(open(gf, "rb").read())
    except OSError:
        h.update(b"no-grammar")
    h.update(os.environ.get("NEWSRAG_MODEL_ID", "qwen15.gguf").encode())
    h.update("metric_kind,currency,figure_source,recordline,enstat,domains,geo,unit,enreach,enheld,rate,enrec,enest,v12".encode())  # v12: en estimated-at parser (GASTAT headline estimates)  # v4: deterministic record-line parser added; stale LLM parses of record lines invalid
    return h.hexdigest()


_WB_CODES = {  # World Bank indicator id -> canonical domain tag
    "SL.UEM.TOTL.ZS": "unemployment",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "NY.GDP.MKTP.CD": "gdp_total",
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "SP.POP.TOTL": "population",
    "SP.DYN.LE00.IN": "life_expectancy",
    "SP.DYN.TFRT.IN": "fertility",
    "IT.NET.USER.ZS": "internet_users",
    "SE.ADT.LITR.ZS": "literacy",
    "NV.AGR.TOTL.ZS": "agriculture_share",
    "EN.ATM.PM25.MC.M3": "pm25_exposure",
    "EN.GHG.CO2.PC.CE.AR5": "co2_per_capita",
    "SP.URB.TOTL.IN.ZS": "urban_population_share",
    "SL.TLF.CACT.FE.ZS": "female_labor_participation",
    "NE.EXP.GNFS.ZS": "exports_share",
    "SP.POP.0014.TO.ZS": "population_0_14_share",
}

# ordered, most specific first: the first domain whose cue appears wins
_DOMAIN_CUES = [
    ("female_labor_participation", ("مشاركة الإناث", "female")),
    ("unemployment", ("بطالة", "unemployment", "labor force", "قوة العمل")),
    ("life_expectancy", ("العمر المتوقع", "life expectancy")),
    ("fertility", ("الخصوبة", "fertility")),
    ("internet_users", ("الإنترنت", "internet")),
    ("literacy", ("محو أمية", "أمية", "literacy")),
    ("agriculture_share", ("الزراعة", "agriculture")),
    ("pm25_exposure", ("pm2.5", "جسيمات")),
    ("co2_per_capita", ("انبعاثات", "co2", "ثاني أكسيد")),
    ("urban_population_share", ("سكان الحضر", "urban population")),
    ("exports_share", ("صادرات", "exports")),
    ("population_0_14_share", ("0-14", "الفئة العمرية")),
    ("gdp_per_capita", ("نصيب الفرد", "per capita")),
    ("gdp_growth", ("نمو الناتج", "النمو السنوي للناتج", "gdp growth", "real gdp growth")),
    ("gdp_total", ("الناتج المحلي", "gdp")),
    ("remittances", ("تحويلات", "remittances")),
    ("interest_rate", ("فائدة", "interest rate", "deposit rate", "lending rate",
                        "discount rate", "السياسة النقدية")),
    ("reserves", ("احتياط", "reserves")),
    ("population", ("عدد السكان", "إجمالي عدد السكان", "population")),
]

_RECORD_PAIR = re.compile(r"([\w.]+): ([^;]+);")
_NULLISH = {"null", "none", "n/a", ""}


def _recordline_facets(text: str) -> dict[str, Any] | None:
    """Deterministic facet parse for machine-rendered record lines
    ("indicator.id: X; indicator.value: Y; date: 2024; value: 2.64; ...").
    The collector itself renders JSON API rows in this shape, so its fields
    are known-exact; asking a 1.5B model to re-read them only drops fields
    (period/indicator lost -> false abstention)."""
    pairs = dict(_RECORD_PAIR.findall(text))
    if "value" not in pairs or "date" not in pairs:
        return None
    val = pairs["value"].strip()
    if val.lower() in _NULLISH or not re.search(r"\d", val):
        return None  # missing data point - honestly empty, not a figure
    indicator = pairs.get("indicator.value", "").strip() or None
    country = pairs.get("country.value", "").strip() or None
    unit = None
    currency = None
    ind_l = (indicator or "").lower()
    if "%" in (indicator or ""):
        unit = "percent"
    elif "(years)" in ind_l:
        unit = "years"
    if "us$" in ind_l or "current us" in ind_l:
        currency = "USD"
    return {"entity": indicator, "measure": indicator,
            "time_period": pairs["date"].strip(),
            "population": country, "attribution": None, "polarity": None,
            "numbers": [val], "unit": unit,
            "metric_kind": None, "currency": currency,
            "figure_source": "official_statement",
            "_wb": _WB_CODES.get(pairs.get("indicator.id", "").strip()),
            "_origin": "record-line-parser"}  # survives cache-hit re-marking


_EN_STAT = re.compile(
    r"([A-Za-z][A-Za-z ]*?) recorded ([\d.]+) percent in ([A-Z][a-z]+ \d{4})",
    re.IGNORECASE)


_EN_REACHED = re.compile(
    r"([A-Za-z][A-Za-z ]*?) reached (?:US\$|\\$)? ?([0-9][0-9,]*(?:\.[0-9]+)?) ?"
    r"(mn|mln|million|bn|billion)? ?at the end of ([A-Z][a-z]+ [0-9]{4})",
    re.IGNORECASE)


def _en_reached_facets(text: str) -> dict[str, Any] | None:
    """Fixed-shape official release: "Net International Reserves reached US$
    57,348.2 mn at the end of September 2026" (CBE). Same rationale as
    _en_stat_facets: parse the known shape, don't ask the 1.5B to translate."""
    m = _EN_REACHED.search(text)
    if not m:
        return None
    measure = " ".join(m.group(1).split())
    if len(measure) < 4:
        return None
    scale = (m.group(3) or "").lower()
    unit = {"mn": "million", "mln": "million", "million": "million",
            "bn": "billion", "billion": "billion"}.get(scale)
    return {"entity": measure, "measure": measure,
            "time_period": m.group(4), "population": None,
            "attribution": None, "polarity": None,
            "numbers": [m.group(2)],
            "metric_kind": None,
            "currency": "USD" if "us$" in text.lower() or "$" in m.group(0) else None,
            "unit": unit,
            "figure_source": "official_statement",
            "_origin": "en-stat-parser"}


_EN_HELD = re.compile(
    r"([A-Za-z][A-Za-z ,]*?) (?:remain|remains|remained|maintained|kept|held|stood)"
    r" (?:unchanged )?(?:also )?(?:maintained )?at ([0-9]+(?:\.[0-9]+)?) percent",
    re.IGNORECASE)


_EN_RECORDED_USD = re.compile(
    r"([A-Za-z][A-Za-z ]*?) recorded (?:about |approximately )?(?:USD|US\$|\$) ?"
    r"([0-9][0-9,]*(?:\.[0-9]+)?) ?(billion|million|bn|mn|mln)? ?"
    r"(?:during|in) (?:the )?([A-Za-z0-9 ]*?[0-9]{4})", re.IGNORECASE)


def _en_recorded_usd_facets(text: str) -> dict[str, Any] | None:
    """"Remittances ... Recorded USD 29.7 Billion During the First Seven Months
    of 2026" (CBE headline shape). The 1.5B invents periods for these
    ("أكتوبر 2025" for a 2026 headline): parse the shape directly."""
    m = _EN_RECORDED_USD.search(text)
    if not m:
        return None
    measure = " ".join(m.group(1).split())
    if len(measure) < 4:
        return None
    scale = (m.group(3) or "").lower()
    unit = {"billion": "billion", "bn": "billion", "million": "million",
            "mn": "million", "mln": "million"}.get(scale)
    return {"entity": measure, "measure": measure,
            "time_period": m.group(4).strip(), "population": None,
            "attribution": None, "polarity": None,
            "numbers": [m.group(2)],
            "metric_kind": None, "currency": "USD", "unit": unit,
            "figure_source": "official_statement",
            "_origin": "en-stat-parser"}


def _en_held_facets(text: str) -> dict[str, Any] | None:
    """"The overnight deposit rate ... remain at 19.00 percent" (CBE MPC).
    A held rate is a level statement: polarity None."""
    m = _EN_HELD.search(text)
    if not m:
        return None
    measure = " ".join(m.group(1).split()).strip(", ")
    if len(measure) < 4:
        return None
    return {"entity": measure, "measure": measure,
            "time_period": None, "population": None,
            "attribution": None, "polarity": None,
            "numbers": [m.group(2)],
            "metric_kind": None, "currency": None, "unit": "percent",
            "figure_source": "official_statement",
            "_origin": "en-stat-parser"}


_EN_ESTIMATED = re.compile(
    r"([A-Za-z][A-Za-z ,']*?) (?:is |are |was |were )?(?:estimated|projected)"
    r" at (?:about |approximately )?([0-9][0-9,.]*) ?"
    r"(million|billion|thousand|mn|bn|mln)? ?"
    r"(?:people|persons|inhabitants|individuals)? ?"
    r"(?:in|as of|by|at) (mid ?- ?\d{4}|[A-Z][a-z]+ \d{4}|\d{4})",
    re.IGNORECASE)


def _en_estimated_facets(text: str) -> dict[str, Any] | None:
    """"The total population of the Kingdom is estimated at 35.3 million
    people in mid -2024" (GASTAT population estimates publication). Headline
    estimates shape; the 1.5B drops the scale word translating it."""
    m = _EN_ESTIMATED.search(text)
    if not m:
        return None
    measure = " ".join(m.group(1).split()).strip(", ")
    if len(measure) < 4:
        return None
    scale = (m.group(3) or "").lower()
    unit = {"million": "million", "mn": "million", "mln": "million",
            "billion": "billion", "bn": "billion",
            "thousand": "thousand"}.get(scale)
    period = re.sub(r"mid ?- ?", "mid-", m.group(4))
    return {"entity": measure, "measure": measure,
            "time_period": period, "population": None,
            "attribution": None, "polarity": None,
            "numbers": [m.group(2)],
            "metric_kind": None, "currency": None, "unit": unit,
            "figure_source": "official_statement",
            "_origin": "en-stat-parser"}


def _en_stat_facets(text: str) -> dict[str, Any] | None:
    """Deterministic parse of English statistical statements of the shape
    "Annual core CPI inflation recorded 14.1 percent in September 2026 ...".
    Official English press releases (CBE et al.) use this fixed shape; the
    1.5B model paraphrases them into degenerate Arabic facets ("النسبة
    المئوية"/"الأسعار"), which the degenerate guard then fails closed.
    Parse the shape directly instead. Figure with no stated direction ->
    polarity None (a level, not a movement claim)."""
    m = _EN_STAT.search(text)
    if not m:
        return None
    measure = " ".join(m.group(1).split())  # e.g. "core CPI inflation"
    if len(measure) < 4:
        return None
    return {"entity": measure, "measure": measure,
            "time_period": m.group(3), "population": None,
            "attribution": None, "polarity": None,
            "numbers": [m.group(2)],
            "metric_kind": None, "currency": None,
            "figure_source": "official_statement",
            "_origin": "en-stat-parser"}


def extract_facets(text: str, generate: Callable[..., str]) -> dict[str, Any] | None:
    _ck = _cache_key(text)
    _cached = _excache_get(_ck)
    if _cached is not None:
        if isinstance(_cached, dict) and _cached.get("_failed"):
            return None  # negative cache: this text already failed extraction
        return {**_cached, "_extraction": "cache-hit"}
    rl = _recordline_facets(text.strip())
    if rl is not None:
        _excache_put(_ck, rl)
        return {**rl, "_extraction": "record-line-parser"}
    en = (_en_stat_facets(text.strip()) or _en_reached_facets(text.strip())
          or _en_held_facets(text.strip()) or _en_recorded_usd_facets(text.strip())
          or _en_estimated_facets(text.strip()))
    if en is not None:
        _excache_put(_ck, en)
        return {**en, "_extraction": "en-stat-parser"}
    """Model-side facet extraction. Caller binds json_schema=FACET_SCHEMA."""
    prompt = _FACET_PROMPT.format(text=text.strip()[:700])
    try:
        obj = _extract_json(str(generate(prompt)))
        if not isinstance(obj, dict):
            obj = _extract_json(str(generate(prompt)))  # one retry: 1.5B emits
            # truncated/unparseable JSON intermittently; a second sample is cheap
            # next to a false "extraction failed" classification (core 14.1 sighting)
    except Exception as exc:
        # generation-side failure (e.g. per-candidate time budget exceeded on
        # slow hardware): classify, negative-cache with the reason, abstain.
        _excache_put(_ck, {"_failed": True, "_reason": f"generate-error:{type(exc).__name__}"})
        return None
    if not isinstance(obj, dict):
        _excache_put(_ck, {"_failed": True})
        return None
    facets = {k: obj.get(k) for k in ("entity", "measure", "time_period", "population", "attribution", "polarity", "numbers", "metric_kind", "currency", "figure_source")}
    # Degenerate-output guard: a measure/entity that carries no scope meaning
    # ("percentage", "the rate") marks a failed extraction; callers abstain
    # rather than compare hollow facets.
    if _norm(facets.get("measure")) in _DEGENERATE or _norm(facets.get("entity")) in _DEGENERATE:
        _excache_put(_ck, {"_failed": True})
        return None
    _excache_put(_ck, facets)
    return {**facets, "_extraction": "fresh"}


_INFL_CUE = ("تضخم", "inflation", "cpi", "أسعار المستهلك", "consumer price")


_COORD_CUE = re.compile(r"منسّ?ق|coordinat|orchestrat", re.IGNORECASE)


def _indicator(f: dict[str, Any]) -> str | None:
    """Canonical indicator tag from entity+measure+population facets. None =
    unknown. The inflation tags require an inflation-domain cue: 'all-republic'
    on a GDP figure is not the all-republic inflation rate."""
    if f.get("_wb"):
        return f["_wb"]  # WB indicator id is itself the canonical tag
    m, pop = _norm(f.get("measure")), _norm(f.get("population"))
    blob = f"{_norm(f.get('entity'))} {m} {pop}"
    if not blob.strip():
        return None
    for tag, cues in _DOMAIN_CUES:
        if any(_norm(c) in blob for c in cues):
            # gdp_per_capita needs the gdp context too; "نصيب الفرد" alone on
            # a non-GDP figure is not the per-capita GDP indicator
            if tag == "gdp_per_capita" and not any(c in blob for c in ("ناتج", "gdp")):
                continue
            return tag
    infl = any(_norm(c) in blob for c in _INFL_CUE)
    if infl and any(a in blob for a in _MEAS_MONTHLY):
        return "monthly"
    if infl and any(a in blob for a in _MEAS_CORE):
        return "core"
    if infl and any(a in blob for a in _POP_URBAN):
        return "urban_headline"
    if infl and any(a in blob for a in _POP_ALLREP):
        return "allrep_headline"
    if infl and ("headline" in blob or "السنوي" in blob or "annual" in blob):
        return "headline_unscoped"
    # Coordination is itself the indicator when the text demands it: a claim
    # about coordinated campaigns is not a volume/accounts/reach metric claim.
    # Without this tag the facet collapses to None and analysis runs scope-less
    # (review: claim indicator dropped to None, caption-grade evidence
    # bridged in through the None-side path). Domain cues above still win.
    if _COORD_CUE.search(blob) or _COORD_CUE.search(_norm(f.get("_sentence"))):
        return "coordination"
    # campaign metric kinds apply only when no domain indicator matched: the
    # 1.5B extractor mislabels ordinary percentage text as campaign volume, so
    # a domain measure (inflation, unemployment...) always wins over the tag.
    mk = _norm(f.get("metric_kind"))
    _CAMP = ("حملة", "حملات", "حساب", "منشور", "تغريد", "campaign", "account",
             "post", "tweet", "ad ", "advertis")
    if mk in ("spend", "accounts", "volume", "reach") and any(w in blob for w in _CAMP):
        return f"campaign_{mk}"
    if mk in ("spend", "accounts", "volume", "reach"):
        # extractor tagged a campaign kind but nothing in the text is about a
        # campaign: a percentage of a price group is not campaign volume.
        return "figure_unscoped"
    return None


_COUNTRY = {  # bilingual + iso3 canonical country names (geography is scope)
    "egypt": "egypt", "مصر": "egypt", "egy": "egypt",
    "saudi arabia": "saudi_arabia", "السعودية": "saudi_arabia", "sau": "saudi_arabia",
    "jordan": "jordan", "الأردن": "jordan", "jor": "jordan",
    "morocco": "morocco", "المغرب": "morocco", "mar": "morocco",
    "tunisia": "tunisia", "تونس": "tunisia", "tun": "tunisia",
}


def _country(f: dict[str, Any]) -> str | None:
    """Canonical country from the population facet, bilingual. None = unstated
    or unknown (unknown is NOT a mismatch - abstention, not veto)."""
    pop = _norm(f.get("population"))
    if not pop:
        return None
    pop = re.sub(r"\s*\([^)]*\)", "", pop)  # "Egypt, Arab Rep." handled below
    for name, canon in _COUNTRY.items():
        if name in pop:
            return canon
    if "arab rep" in pop and "egypt" in pop:
        return "egypt"
    return None


_SCOPE_PATH = ["indicator"]  # per-call marker: how scope was established


def compare_facets(claim_f: dict[str, Any], support_f: dict[str, Any]) -> dict[str, Any]:
    """Deterministic scope comparison. Abstains on missing facets; 'contradicts'
    only for same-scope incompatible assertions; scope mismatch is never a
    contradiction (a republisher is not a liar). Results carry scope_path:
    'indicator' = verified domain-indicator match; 'lexical_fallback' = weak
    stemmed-token overlap when no domain indicator exists on either side -
    answers via lexical_fallback are provisional, not verified-scope."""
    _SCOPE_PATH[0] = "indicator"
    ci, si = _indicator(claim_f), _indicator(support_f)
    # figure_unscoped is extractor noise (a campaign tag with no campaign in
    # the text), not a scope statement: treat it as unknown in matching.
    ci = None if ci == "figure_unscoped" else ci
    si = None if si == "figure_unscoped" else si
    if not ci and not si:
        # no domain indicator on either side (generic questions, e.g. "did
        # investigations document a coordinated campaign"): fall back to
        # bilingual stemmed token overlap over entity/measure/population.
        # Two shared scope tokens = same-scope; fewer = abstain.
        from .discover import _tokens
        c_scope = _tokens(" ".join(str(claim_f.get(k) or "") for k in ("entity", "measure", "population"))
                          + " " + str(claim_f.get("_sentence") or ""))
        s_scope = _tokens(" ".join(str(support_f.get(k) or "") for k in ("entity", "measure", "population"))
                          + " " + str(support_f.get("_sentence") or ""))
        shared = c_scope & s_scope
        if len(shared) < 2:
            return {"relation": "insufficient", "dimension": None, "reason_code": "indicator_missing",
                    "reason": f"no domain indicator and weak scope overlap ({sorted(shared)[:4]}); abstaining"}
        _SCOPE_PATH[0] = "lexical_fallback"
    elif not ci or not si:
        # one-sided bridge: the indicator-less side may still be ABOUT a
        # campaign (cue words in its entity/measure/population) - then a
        # campaign_* evidence indicator is scope-compatible, not a mismatch.
        none_side, other = (claim_f, si) if not ci else (support_f, ci)
        blob_n = _norm(" ".join(str(none_side.get(k) or "") for k in ("entity", "measure", "population")))
        _CUE = ("حملة", "حملات", "حساب", "منشور", "تغريد", "campaign", "account", "post", "tweet")
        if other.startswith("campaign_") and any(w in blob_n for w in _CUE):
            _SCOPE_PATH[0] = "cue_bridge"
        if not (other.startswith("campaign_") and any(w in blob_n for w in _CUE)):
            return {"relation": "insufficient", "dimension": None, "reason_code": "indicator_missing",
                    "reason": "indicator facet missing on one side; abstaining instead of default-matching"}
    if ci != si or "unscoped" in (ci, si):
        dim = "measure" if ("core" in (ci, si) or ci != si) else "population"
        return {"relation": "different_scope", "dimension": dim, "reason_code": "indicator_differs",
                "reason": f"indicator differs: claim={ci} vs evidence={si}"}
    # qualifier conflicts in the measure strings veto same-scope even when the
    # indicator fallback passed: monthly is not annual; an index level is not a
    # rate. Without this, weak-overlap matching binds the wrong figure.
    cm, sm = _norm(claim_f.get("measure")), _norm(support_f.get("measure"))
    for a, b in (("سنوي", "شهري"), ("annual", "monthly"),
                 ("معدل", "رقم قياسي"), ("rate", "index"), ("نسبة", "رقم قياسي")):
        if (a in cm and b in sm) or (b in cm and a in sm):
            return {"relation": "different_scope", "dimension": "measure",
                    "reason_code": "qualifier_conflict",
                    "reason": f"measure qualifier conflict: claim={claim_f.get('measure')!r} vs evidence={support_f.get('measure')!r}"}
    # rate-type qualifiers distinguish same-domain instruments: the overnight
    # DEPOSIT rate is not the DISCOUNT rate (19.00 vs 19.50 both "held")
    _RATE_TYPES = (("deposit", "إيداع", "الايداع"), ("discount", "خصم"),
                   ("lending", "إقراض", "الاقراض"))
    def _rtype(blob):
        return {i for i, names in enumerate(_RATE_TYPES)
                if any(_norm(n) in blob for n in names)}
    cm_b, sm_b = _norm(claim_f.get("measure")) + " " + _norm(claim_f.get("entity")), \
                 _norm(support_f.get("measure")) + " " + _norm(support_f.get("entity"))
    crt, srt = _rtype(cm_b), _rtype(sm_b)
    if crt and srt and not (crt & srt):
        return {"relation": "different_scope", "dimension": "measure",
                "reason_code": "rate_type_differs",
                "reason": f"rate instrument differs: claim={claim_f.get('measure')!r} vs evidence={support_f.get('measure')!r}"}
    ct, st = _period(claim_f.get("time_period")), _period(support_f.get("time_period"))
    if not ct or not st:
        return {"relation": "insufficient", "dimension": None, "reason_code": "period_missing",
                "reason": "time period missing on one side; abstaining"}
    cy, sy = {y for _, y in ct if y}, {y for _, y in st if y}
    if cy and sy and not (cy & sy):
        return {"relation": "different_scope", "dimension": "time", "reason_code": "year_differs",
                "reason": f"year differs: claim={claim_f.get('time_period')!r} vs evidence={support_f.get('time_period')!r}"}
    if bool(cy) != bool(sy):
        return {"relation": "insufficient", "dimension": None, "reason_code": "year_one_sided",
                "reason": "year stated on one side only; cannot confirm the period, abstaining"}
    cm, sm = {m for m, _ in ct if m}, {m for m, _ in st if m}
    if cm and not sm:
        # claim names a month, evidence gives only the year: cannot confirm the
        # month - abstain (review: month-0 is "unspecified", not a month)
        return {"relation": "insufficient", "dimension": None, "reason_code": "month_one_sided",
                "reason": "claim names a month but evidence states only the year; abstaining"}
    if cm and sm and not (cm & sm):
        return {"relation": "different_scope", "dimension": "time", "reason_code": "period_differs",
                "reason": f"period differs: claim={claim_f.get('time_period')!r} vs evidence={support_f.get('time_period')!r}"}
    # year-only claim with month-specific evidence in the same year is
    # containment (June 2025 within 2025), not a period difference.
    # geography is scope: an Egypt figure never answers a Saudi question.
    ccy, scy = _country(claim_f), _country(support_f)
    if ccy and scy and ccy != scy:
        return {"relation": "different_scope", "dimension": "geography",
                "reason_code": "country_differs",
                "reason": f"country differs: claim={claim_f.get('population')!r} vs evidence={support_f.get('population')!r}"}
    # spend comparisons need the same currency: $167,000 is not 167,000 riyals.
    if ci == si == "campaign_spend":
        cc, sc = _norm(claim_f.get("currency")), _norm(support_f.get("currency"))
        if cc and sc and cc != sc:
            return {"relation": "different_scope", "dimension": "measure", "reason_code": "currency_differs",
                    "reason": f"spend currency differs: claim={claim_f.get('currency')!r} vs evidence={support_f.get('currency')!r}"}
    # attribution: compared only when BOTH sides attribute; a mismatch means the
    # evidence does not verify THAT speaker - different_scope, never contradiction,
    # and it is checked BEFORE direction/figures so a republisher or a different
    # speaker can never be scored as contradicting (a republisher is not a liar).
    # generic framing words are not attributions: "according to investigations"
    # in a question does not name a speaker the evidence must match.
    _GEN_ATTR = ("تحقيق", "تحقيقات", "تقارير", "صحف", "صحفي", "مصادر", "وسائل",
                 "investigation", "report", "media", "press", "source")
    def _attr_or_none(f):
        a = _norm(f.get("attribution"))
        if not a or any(g in a for g in _GEN_ATTR):
            return None
        return f.get("attribution")
    ca = _canon(_attr_or_none(claim_f), (_ATTR_CBE, _ATTR_CAPMAS))
    sa = _canon(_attr_or_none(support_f), (_ATTR_CBE, _ATTR_CAPMAS))
    attribution_mismatch = bool(ca and sa and ca != sa)
    if attribution_mismatch:
        return {"relation": "different_scope", "dimension": "attribution", "reason_code": "attribution_differs",
                "reason": f"attribution differs: claim={claim_f.get('attribution')!r} vs evidence={support_f.get('attribution')!r}; the evidence does not verify that speaker"}
    cp, sp = claim_f.get("polarity"), support_f.get("polarity")
    opposite = cp in ("up", "down") and sp in ("up", "down") and cp != sp
    if opposite:
        return {"relation": "contradicts", "dimension": "direction", "reason_code": "opposite_direction",
                "reason": f"same scope/period, opposite direction: claim={cp}, evidence={sp}"}
    cn, sn = _numbers(claim_f.get("numbers")), _numbers(support_f.get("numbers"))
    if not cn or not sn:
        return {"relation": "insufficient", "dimension": None, "reason_code": "figures_missing",
                "reason": "one side states no figures; cannot settle numerically"}
    numbers_agree = bool(cn & sn)
    if not numbers_agree:
        return {"relation": "contradicts", "dimension": None, "reason_code": "figures_incompatible",
                "reason": f"same scope/period, incompatible figures: claim={sorted(cn)} vs evidence={sorted(sn)}"}
    return {"relation": "supports", "dimension": None, "reason_code": "scope_agree",
            "reason": "same indicator and period, figures agree, no incompatible direction or attribution"}
