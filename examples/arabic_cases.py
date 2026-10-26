"""Documented Arabic example cases: expected vs actual, with trace reasons.

Fixture-based, no network, no live model: the generator is scripted per case so
results are deterministic and reproducible. Sources are SYNTHETIC fixtures in
realistic journalistic Arabic; "mutation" cases alter a number/unit/negation
against the fixture source and are labeled ADVERSARIAL MUTATIONS - they are not
claims about real-world occurrences.

This is the DEV case set (design + debugging). It is not the frozen eval set;
do not tune against frozen data. Each case records the expected decision, the
actual decision, and the decisive trace step so a reviewer can see WHY.
"""
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, 'src')
from newsrag import NewsroomRAG, Source
from newsrag.assist import assist

D = datetime(2026, 9, 3, tzinfo=timezone.utc)

def src(i, text, title="مصدر"):
    return Source(id=i, title=title, text=text, url=f"https://example.org/{i}", published_at=D)

FISH = "بلغ عدد الصيادين المتضررين 225 صيادًا وفق التقرير الرسمي. هذه جملة إضافية طويلة بما يكفي للاسترجاع والفحص."
HEAT = "ارتفعت درجات حرارة البحر في المنطقة خلال عام 2023 بشكل قياسي وفق المرصد البحري. تفاصيل إضافية طويلة بما يكفي للفحص."
HEAT_B = "قال المرصد البحري إن درجات حرارة البحر سجلت ارتفاعًا قياسيًا خلال عام 2024 في المنطقة. تفاصيل أخرى طويلة بما يكفي."

def gen(draft):
    def _g(p):
        if "Rewrite the ANSWER" in p: return '{"answer":"x"}'
        if "Extract up to 5 checkable" in p: return '{"claims":[]}'
        if "skeptical reviewer" in p: return '{"contradiction":0.1}'
        return draft
    return _g

def draft(claim, answer=None):
    return json.dumps({"answer": answer or claim + ".", "claims": [claim]}, ensure_ascii=False)

CASES = [
    {"id": "num-mutation-western", "kind": "adversarial_mutation", "category": "numbers",
     "note": "claim number 900 mutated vs source 225 - not a real-world claim",
     "sources": [src("s1", FISH)], "question": "كم عدد الصيادين المتضررين؟",
     "draft": draft("بلغ عدد الصيادين المتضررين 900 صيادًا"),
     "expected": "insufficient", "expect_reason": "numbers_not_in_support"},
    {"id": "num-mutation-eastern", "kind": "adversarial_mutation", "category": "numbers",
     "note": "Eastern digits ٩٠٠ mutated vs source 225 - normalization must catch it",
     "sources": [src("s1", FISH)], "question": "كم عدد الصيادين المتضررين؟",
     "draft": draft("بلغ عدد الصيادين المتضررين ٩٠٠ صيادًا"),
     "expected": "insufficient", "expect_reason": "numbers_not_in_support"},
    {"id": "num-supported", "kind": "authentic_fixture", "category": "numbers",
     "note": "claim matches the fixture source number",
     "sources": [src("s1", FISH)], "question": "كم عدد الصيادين المتضررين؟",
     "draft": draft("بلغ عدد الصيادين المتضررين 225 صيادًا"),
     "expected": "answer", "expect_reason": "claims_supported"},
    {"id": "unit-mismatch", "kind": "adversarial_mutation", "category": "units",
     "note": "unit mutated km->miles with the SAME number - documents the open role/unit gap",
     "sources": [src("s1", "امتدت المنطقة المتضررة 225 كيلومترًا وفق التقرير الرسمي. تفاصيل إضافية طويلة بما يكفي للفحص والاسترجاع.")],
     "question": "ما مساحة المنطقة المتضررة؟",
     "draft": draft("امتدت المنطقة المتضررة 225 ميلًا وفق التقرير"),
     "expected": "insufficient", "expect_reason": "unsupported_claims",
     "note2": "FIXED by the deterministic unit lexicon guard: both sides name units and share none "
              "(كيلومتر vs ميل) -> unit_mismatch fail-closed. Bounded by the lexicon; unlisted units "
              "remain unchecked (open gap)."},
    {"id": "date-mutation", "kind": "adversarial_mutation", "category": "dates",
     "note": "year mutated 2023->2024 - caught by the number guard",
     "sources": [src("s1", HEAT)], "question": "ماذا حدث لدرجات حرارة البحر؟",
     "draft": draft("ارتفعت درجات حرارة البحر في المنطقة خلال عام 2024 بشكل قياسي"),
     "expected": "insufficient", "expect_reason": "numbers_not_in_support"},
    {"id": "negation-mismatch", "kind": "adversarial_mutation", "category": "negation",
     "note": "claim negated vs affirmative source - negation parity must catch it",
     "sources": [src("s1", HEAT)], "question": "ماذا حدث لدرجات حرارة البحر؟",
     "draft": draft("لم ترتفع درجات حرارة البحر في المنطقة خلال عام 2023 بشكل قياسي"),
     "expected": "insufficient", "expect_reason": "negation_mismatch"},
    {"id": "paraphrase-same-facts", "kind": "authentic_fixture", "category": "paraphrase",
     "note": "paraphrased question, same facts - retrieval finds NO passage at all",
     "sources": [src("s1", FISH)], "question": "ما حجم الضرر الذي لحق بالصيادين؟",
     "draft": draft("أشار التقرير الرسمي إلى تضرر 225 من الصيادين"),
     "expected": "insufficient", "expect_reason": "unsupported_claims",
     "known_limitation": "lexical retrieval recall: the paraphrased question shares too few surface "
                         "tokens with the source (no stemming/embeddings in the default path), so zero "
                         "passages are retrieved. Fail-closed output is safe but unhelpful here - "
                         "the documented Arabic retrieval fragility class."},
    {"id": "conflicting-sources", "kind": "authentic_fixture", "category": "conflict",
     "note": "two fixture sources disagree on the year; claim follows one - trace must name the ref used",
     "sources": [src("s1", HEAT, "المرصد أ"), src("s2", HEAT_B, "المرصد ب")],
     "question": "في أي عام سجل الارتفاع القياسي؟",
     "draft": draft("سجل المرصد ارتفاعًا قياسيًا لدرجات حرارة البحر خلال عام 2023"),
     "expected": "answer", "expect_reason": "claims_supported",
     "note2": "FIXED by deeper candidate search: the top-5 candidates are evaluated and the "
              "highest-ranked sentence passing ALL guards is accepted, so the wrong-year sentence "
              "fails its guards and the correct source-A sentence is found. Guards unchanged. "
              "Cross-source disagreement surfacing remains open (no contradiction merge)."},
]

def run_case(c):
    rag = NewsroomRAG(generate=gen(c["draft"]))
    rag.add(*c["sources"])
    r = assist(rag, c["question"])
    claim_steps = [t for t in r.trace if t.name == "claim_support"]
    decisive = None
    for t in claim_steps:
        if t.outcome == "fail":
            decisive = {"failed_checks": t.detail.get("failed_checks", []),
                        "numbers_missing": t.detail.get("numbers_missing", []),
                        "similarity": t.detail.get("similarity"),
                        "negation_parity": t.detail.get("negation_parity")}
            break
    if decisive is None and claim_steps:
        t = claim_steps[0]
        decisive = {"evidence_ref": t.detail.get("evidence_ref"), "offset": t.detail.get("offset"),
                    "similarity": t.detail.get("similarity")}
    actual_reason = r.insufficient_reason or "claims_supported"
    failed_checks = (decisive or {}).get("failed_checks", [])
    ok = r.decision == c["expected"] and (
        c["expect_reason"] == actual_reason or c["expect_reason"] in failed_checks)
    return {"id": c["id"], "kind": c["kind"], "category": c["category"], "note": c["note"],
            "known_limitation": c.get("known_limitation"), "note2": c.get("note2"),
            "expected": c["expected"], "actual": r.decision, "match": ok,
            "reason": actual_reason, "decisive_trace": decisive, "set": "dev"}

def main():
    results = [run_case(c) for c in CASES]
    for r in results:
        flag = "OK " if r["match"] else "MISMATCH"
        lim = " [KNOWN LIMITATION]" if r.get("known_limitation") else ""
        print(f'{flag} {r["id"]}: expected={r["expected"]} actual={r["actual"]} reason={r["reason"]}{lim}')
    json.dump(results, open('/tmp/newsrag-data/arabic_cases_dev.json', 'w'), ensure_ascii=False, indent=2)
    n_ok = sum(1 for r in results if r["match"])
    print(f'{n_ok}/{len(results)} cases behave as documented')

if __name__ == "__main__":
    main()
