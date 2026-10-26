"""Journalistic fact-check brief rendering for AssistResult.

Deterministic, model-free presentation layer: separates what passed the
library's checks (facts, with source refs) from what did not (unsupported or
withheld material) and states the limits of the method. The wording is
deliberately bounded: the checks are automated lexical checks with negation,
number and unit guards - not entailment and not human editorial review.
"""

from __future__ import annotations

from .assist import AssistResult

_DECISION_AR = {
    "answer": "إجابة اجتازت كل جملها فحص السند الآلي",
    "extractive": "مقتطفات موثقة فقط من نص المصدر",
    "insufficient": "الأدلة المسترجعة غير كافية للإجابة",
}

_REASON_AR = {
    "draft_unusable": "تعذر توليد مسودة صالحة (خلل توليد، لا دليل على السؤال)",
    "no_checkable_claims": "لم تُستخرج ادعاءات قابلة للفحص",
    "unsupported_claims": "ادعاءات بلا سند كافٍ في المقاطع المسترجعة",
}

_LIMITS_AR = ("حدود الفحص: تطابق لفظي آلي مع ضوابط النفي والأرقام والوحدات؛ "
              "ليس استلزامًا لغويًا ولا مراجعة تحريرية بشرية، ولا يثبت شيئًا خارج "
              "نطاق المقاطع المسترجعة.")


def format_brief(result: AssistResult) -> str:
    """Render an AssistResult as an Arabic fact-check brief.

    Sections: الخلاصة (verdict line), ما تحقق (supported claims with refs),
    ما لم يتحقق (unsupported/withheld), المصادر, حدود الفحص. Never upgrades
    the decision and never hides withheld material.
    """
    r = result
    lines: list[str] = []
    verdict = _DECISION_AR.get(r.decision, r.decision)
    if r.insufficient_reason:
        verdict += " — " + _REASON_AR.get(r.insufficient_reason, r.insufficient_reason)
    lines.append("الخلاصة: " + verdict)

    supported = [c for c in r.claims if c.supported]
    unsupported = [c for c in r.claims if not c.supported]
    if supported:
        lines.append("")
        lines.append("ما تحقق (سند لفظي من المقاطع):")
        for c in supported:
            ref = f" [{c.evidence_ref}]" if c.evidence_ref else ""
            lines.append(f"- {c.text}{ref}")
    if unsupported:
        lines.append("")
        lines.append("ما لم يتحقق (استُبعد من أي إخراج معروض):")
        for c in unsupported:
            lines.append(f"- {c.text}")
    if r.decision != "insufficient" and r.text:
        lines.append("")
        lines.append("النص المعروض:")
        lines.append(r.text)
    if r.evidence:
        lines.append("")
        lines.append("المصادر المسترجعة:")
        seen = set()
        for e in r.evidence:
            if e.ref in seen:
                continue
            seen.add(e.ref)
            url = f" — {e.url}" if e.url else ""
            lines.append(f"- [{e.ref}] {e.title}{url}")
    if r.warnings:
        lines.append("")
        lines.append("تنبيهات:")
        for w in r.warnings:
            lines.append(f"- {w}")
    lines.append("")
    lines.append(_LIMITS_AR)
    return "\n".join(lines)
