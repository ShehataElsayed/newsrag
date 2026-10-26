"""v1.3 grounding+malformed suite: bind REQUIRES non-null verified spans.claim
(auditor hole closure). v1.2 malformed cases retained; case expectations updated where
v1.3 deliberately changes the outcome (empty/zero claim now abstains, never binds)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as h

DOC = "The network ran pages. Citizen Lab said the group used bots."
CASES = [
    # v1.3 hole-closure regressions (zero/empty/ungrounded claim must NOT bind)
    ("zero spans outlet_analysis",  '{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","uncertainty":false,"contradiction":false,"spans":{}}', "abstain", "claim_span_missing"),
    ("empty-list spans coerce",     '{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","uncertainty":false,"contradiction":false,"spans":[]}', "abstain", "claim_span_missing"),
    ("null claim span",             '{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","uncertainty":false,"contradiction":false,"spans":{"claim":null}}', "abstain", "claim_span_missing"),
    ("unverified claim text",       '{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","uncertainty":false,"contradiction":false,"spans":{"claim":"not in the document at all"}}', "abstain", "model_invalid:span_unverified:claim"),
    ("document binds WITH claim",   '{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"document","uncertainty":false,"contradiction":false,"spans":{"claim":"The network ran pages.","attribution":"Citizen Lab"}}', "bind", "eligible:document"),
    ("outlet binds WITH claim",     '{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","uncertainty":false,"contradiction":false,"spans":{"claim":"The network ran pages."}}', "bind", "eligible:outlet_analysis"),
    # v1.2 malformed-shape cases retained (unchanged expectations)
    ("nested-list role",            '{"role":[["finding"],"denial"],"subject_kind":"analysis_subject","asserted_by":null,"uncertainty":false,"contradiction":false,"spans":{}}', "abstain", "model_invalid:bad_role_set"),
    ("dict role element",           '{"role":[{"x":1}],"subject_kind":"analysis_subject","asserted_by":null,"uncertainty":false,"contradiction":false,"spans":{}}', "abstain", "model_invalid:bad_role_set"),
    ("list subject_kind",           '{"role":["finding"],"subject_kind":["analysis_subject"],"asserted_by":null,"uncertainty":false,"contradiction":false,"spans":{}}', "abstain", "model_invalid:bad_subject"),
    ("list asserted_by",            '{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":["document"],"uncertainty":false,"contradiction":false,"spans":{}}', "abstain", "model_invalid:bad_asserted_by"),
    ("spans non-empty list",        '{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"document","uncertainty":false,"contradiction":false,"spans":["x"]}', "abstain", "model_invalid:bad_spans"),
    ("top-level list",              '[1,2,3]', "abstain", "model_invalid:not_object"),
    ("top-level string",            '"hello"', "abstain", "model_invalid:not_object"),
]
fails = 0
for name, out, wd, wr in CASES:
    g = h.grade_unit(out, DOC, 0, 10)
    ok = g["decision"] == wd and g.get("reason") == wr
    if not ok: fails += 1
    print(("PASS" if ok else "FAIL"), name, "->", g["decision"], g.get("reason"))
print("v1.3 grounding+malformed suite: %d / %d" % (len(CASES) - fails, len(CASES)))
sys.exit(1 if fails else 0)
