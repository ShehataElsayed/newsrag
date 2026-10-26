"""Narrative assembly: findings -> a sourced Arabic investigation draft.

Multi-axis structure: each decomposed claim is an axis; the draft presents per
-axis what the evidence settled, what was related but out of scope, and what
stayed open - then limits and the source registry. Every figure is copied
from a finding (never generated); every paragraph carries its provenance.
Answers established through weak lexical scope matching are marked provisional
in the text itself. This is a sourced technical draft, not finished journalism
- that classification is stated in the header.
"""
from __future__ import annotations
import re

from typing import Any


_SCALE_AR = {"million": "مليون", "billion": "مليار"}
_MONTH_EN_AR = {"january": "يناير", "february": "فبراير", "march": "مارس",
                "april": "أبريل", "may": "مايو", "june": "يونيو", "july": "يوليو",
                "august": "أغسطس", "september": "سبتمبر", "october": "أكتوبر",
                "november": "نوفمبر", "december": "ديسمبر"}


def _period_ar(p: str) -> str:
    """Render an English period phrase in Arabic when its shape is known;
    unknown shapes return unchanged (the caller then prefers the claim side)."""
    t = str(p or "").strip()
    m = re.fullmatch(r"([A-Za-z]+) (\d{4})", t)
    if m and m.group(1).lower() in _MONTH_EN_AR:
        return f"{_MONTH_EN_AR[m.group(1).lower()]} {m.group(2)}"
    m = re.fullmatch(r"(?:The )?First ([A-Za-z]+) Months of (\d{4})", t, re.IGNORECASE)
    if m:
        _N = {"three": "الثلاثة", "four": "الأربعة", "five": "الخمسة",
              "six": "الستة", "seven": "السبعة", "eight": "الثمانية",
              "nine": "التسعة", "ten": "العشرة", "eleven": "الأحد عشر"}
        word = m.group(1).lower()
        if word in _N:
            return f"الأشهر {_N[word]} الأولى من {m.group(2)}"
    m = re.fullmatch(r"mid-(\d{4})", t, re.IGNORECASE)
    if m:
        return f"منتصف {m.group(1)}"
    m = re.fullmatch(r"(\d{1,2}) ([A-Za-z]+) (\d{4})", t)
    if m and m.group(2).lower() in _MONTH_EN_AR:
        return f"{m.group(1)} {_MONTH_EN_AR[m.group(2).lower()]} {m.group(3)}"
    return t


_PLACES = ("مصر", "السعودية", "الأردن", "المغرب", "تونس", "الجمهورية", "الحضر",
           "المدن", "الريف", "egypt", "saudi", "jordan", "morocco", "tunisia")

_CUR_AR = {"USD": "دولار أمريكي", "EGP": "جنيه مصري", "SAR": "ريال سعودي",
           "AED": "درهم إماراتي", "EUR": "يورو", "JOD": "دينار أردني",
           "MAD": "درهم مغربي", "TND": "دينار تونسي"}


def _group(a: str) -> str:
    """Thousands grouping at the figure's own stated decimals (no precision
    invention: 3338.47371043679 renders grouped but untruncated)."""
    try:
        intpart, _, frac = a.partition(".")
        grouped = f"{int(intpart):,}"
        return f"{grouped}.{frac}" if frac else grouped
    except (ValueError, OverflowError):
        return a


_REC_LINE = re.compile(r"^[\w.]+: [^;]+; ")


def _quote(text: str, limit: int = 180) -> str:
    """Quote evidence readably: collector record lines as key: value pairs
    (never clipped mid-pair), prose clipped at a word boundary."""
    text = (text or "").strip()
    if _REC_LINE.match(text):
        pairs = [p.strip() for p in text.split(";") if p.strip()]
        out = ""
        for pr in pairs:
            cand = f"{out}؛ {pr}" if out else pr
            if len(cand) > limit:
                break
            out = cand
        return out or text[:limit]
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _render_answer(f: dict[str, Any]) -> str:
    vals = []
    for a in (f.get("answer") or []):
        disp = a
        if re.fullmatch(r"-?\d+\.\d{3,}", str(a)):
            # raw API precision (2.64893068308965) is not a readable figure:
            # render at 2 decimals; the card keeps the raw bound value
            disp = f"{float(a):.2f}".rstrip("0").rstrip(".")
        if (f"{a}%" in f["evidence"] or f"{a} %" in f["evidence"]
                or f"{a} percent" in f["evidence"] or f"{a}٪" in f["evidence"]
                or (f.get("answer_context") or {}).get("unit") == "percent"):
            vals.append(f"{_group(disp)}٪")
            continue
        ctx0 = f.get("answer_context") or {}
        cur = ctx0.get("currency")
        if cur:
            scale = _SCALE_AR.get(str(ctx0.get("unit") or "").lower())
            mid = f" {scale}" if scale else ""
            vals.append(f"{_group(disp)}{mid} {_CUR_AR.get(str(cur).upper(), str(cur))}")
        elif (f.get("answer_context") or {}).get("unit") == "years":
            vals.append(f"{_group(disp)} سنة")
        elif str(ctx0.get("unit") or "").lower() in _SCALE_AR:
            # scale word without a currency (population counts, "35.3 million
            # people"). Guard: a figure already expanded by sentence-level
            # normalization ("35 million" -> 35000000) must NOT get the scale
            # word again (35,000,000 مليون would be a 10^6 magnitude error);
            # expanded counts are >= 1000, scale-unit figures are not.
            try:
                small = abs(float(str(a).replace(",", ""))) < 1000
            except ValueError:
                small = False
            scale = _SCALE_AR[str(ctx0["unit"]).lower()]
            vals.append(f"{_group(disp)} {scale}" if small else _group(disp))
        else:
            vals.append(_group(disp))
    val = "، ".join(vals)
    ctx = f.get("answer_context") or {}
    lead = f"أفاد مصدر «{f['source']}»" + (f" ({f.get('publisher')})" if f.get("publisher") else "")
    if val and ctx.get("measure"):
        pop = ctx.get("population") or ""
        if pop and not any('\u0600' <= ch <= '\u06FF' for ch in str(pop)):
            pop = ""  # latin scope fragment = extraction artifact, not a renderable label
        pop_tok = set(re.findall(r"[A-Za-z؀-ۿ]+", str(pop)))
        prep = "في" if pop_tok & set(_PLACES) else "لدى"  # المصريون are people, not a place
        where = f" {prep} {pop}" if pop else ""
        tp = _period_ar(ctx.get("time_period")) if ctx.get("time_period") else ""
        if tp and not any('\u0600' <= ch <= '\u06FF' for ch in tp):
            yr = re.findall(r"20\d\d", tp)
            tp = yr[0] if yr else ""  # untranslatable latin period -> year only
        when = f" عن {tp}" if tp else ""
        lead += f" بأن مؤشر «{ctx['measure']}»{where} بلغ {val}{when}"
    elif val:
        lead += f" بأن القيمة هي {val} (الوحدة/المؤشر/الفترة غير مستخرجة - سُجلت كقيد)"
    if f.get("scope_path") in ("lexical_fallback", "cue_bridge"):
        lead += " [مطابقة نطاق معجمية - نتيجة أولية تحتاج تحققاً أقوى]"
    return f"{lead}، ونص الدليل: «{_quote(f['evidence'], 300)}»."


def assemble_narrative(question: str, result: dict[str, Any]) -> str:
    parts: list[str] = [
        f"السؤال البحثي: {question.strip()}",
        "(مسودة تقنية موثقة بالمصادر - مرحلة تحقق آلي، ليست تحقيقاً صحفياً نهائياً)",
        ""]
    claims = result.get("decomposed_claims", [])
    findings = result.get("findings", [])
    multi = len(claims) > 1

    def axis_block(idx: int, claim_text: str, fs: list[dict[str, Any]]) -> None:
        if multi:
            parts.append(f"المحور {idx}: {claim_text}")
        answers = [f for f in fs if f.get("relation") in ("answers", "supports")]
        if answers:
            parts.append("ما توصلت إليه المراجعة:")
            seen = set()
            for f in sorted(answers, key=lambda f2: -len(f2.get("evidence", ""))):
                key = (f["source"], tuple(sorted(f.get("answer") or [])),
                       # qualitative bindings carry no figures: distinguish by evidence
                       "" if f.get("answer") else (f.get("evidence") or "")[:80])
                if key in seen:
                    continue  # same figure from the same source (headline + nav): once
                seen.add(key)
                parts.append(_render_answer(f))
        else:
            parts.append("لم تتوصل المراجعة إلى إجابة موثقة لهذا المحور بالمصادر المتاحة." if multi
                         else "لم تتوصل المراجعة إلى إجابة موثقة بالمصادر المتاحة.")
        out_scope = [f for f in fs if f.get("relation") == "different_scope"]
        if out_scope:
            parts.append("أدلة ذات صلة خارج نطاق السؤال (سُجلت ولم تُحتسب إجابة):")
            for f in out_scope[:4]:
                parts.append(f"- من «{f['source']}»: «{_quote(f['evidence'])}» — السبب: {f.get('reason', '')}.")
        insuff = [f for f in fs if f.get("relation") == "insufficient"]
        if insuff:
            parts.append("أدلة تعذّر استيفاؤها (سُجلت ولم تُحتسب):")
            for f in insuff[:4]:
                parts.append(f"- من «{f['source']}»: «{_quote(f['evidence'])}» — السبب: {f.get('reason', '')}.")
        if multi:
            parts.append("")

    if multi:
        for i, c in enumerate(claims, 1):
            fs = [f for f in findings if f.get("claim") == c["claim"]]
            axis_block(i, c["claim"], fs)
        parts.append("الخلاصة عبر المحاور")
        _PROVISIONAL_PATHS = ("lexical_fallback", "cue_bridge")

        def _has_binding(c: dict[str, Any], provisional: bool) -> bool:
            return any(f.get("claim") == c["claim"] and f.get("relation") in ("answers", "supports")
                       and ((f.get("scope_path") in _PROVISIONAL_PATHS) == provisional)
                       for f in findings)
        # An axis counts as settled only on verified-scope bindings; provisional
        # lexical/cue-bridge bindings document the axis partially, never settle it.
        settled = sum(1 for c in claims if _has_binding(c, provisional=False))
        partial = sum(1 for c in claims
                      if not _has_binding(c, provisional=False) and _has_binding(c, provisional=True))
        line = f"سُوِّي {settled} من {len(claims)} محاور بأدلة موثقة"
        if partial:
            line += f"، ووُثّق {partial} منها توثيقاً جزئياً بنتائج أولية تحتاج تحققاً أقوى"
        settle_sources = {f.get("source") for c in claims for f in findings
                          if f.get("claim") == c["claim"] and f.get("relation") in ("answers", "supports")
                          and f.get("scope_path") not in _PROVISIONAL_PATHS}
        if settled and len(settle_sources) == 1:
            # one investigation's analysis is not independent multi-source
            # verification (review)
            line += "، وفق تحليل مصدر واحد لا تحققاً من مصادر مستقلة متعددة"
        parts.append(line + "؛ التفاصيل والمصادر في كل محور أعلاه.")
        parts.append("")
    else:
        axis_block(1, claims[0]["claim"] if claims else question, findings)
        one_src = {f.get("source") for f in findings if f.get("relation") in ("answers", "supports")}
        if one_src and len(one_src) == 1:
            parts.append("ملاحظة منهجية: النتائج أعلاه وفق تحليل مصدر واحد، وليست تحققاً من مصادر مستقلة متعددة.")
        parts.append("")

    gaps = result.get("gaps", [])
    provisional = [f for f in findings if f.get("relation") in ("answers", "supports")
                   and f.get("scope_path") in ("lexical_fallback", "cue_bridge")]
    parts.append("حدود هذه المراجعة" if multi else "الثغرات وما تعذّر التحقق منه")
    for g in gaps:
        parts.append(f"- ({g.get('stage')}) {g.get('reason', '')}")
    for f in provisional:
        parts.append(f"- نتيجة أولية بمطابقة معجمية ضعيفة النطاق تحتاج تحققاً أقوى: «{f['evidence'][:120]}»")
    if not gaps and not provisional:
        parts.append("- لا ثغرات مسجلة في حدود المصادر المستخدمة.")
    parts.append("")
    prov = result.get("provenance", [])
    if prov:
        parts.append("سجل المصادر")
        for p in prov:
            parts.append(f"- {p.get('source')} — الناشر: {p.get('publisher', 'غير محدد')} — الفترة المرجعية: {p.get('reference_period', 'غير محددة')} — {p.get('url', '')} — وقت الجلب: {p.get('fetched_at', '')} — الحالة: {p.get('status', '')}")
    return "\n".join(parts)
