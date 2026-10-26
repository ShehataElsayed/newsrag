import os
import sys; sys.path.insert(0, '/tmp/phase1')
import harness as h
doc = "# شبكة حسابات موالية\n\nقالت وزارة الصحة إن الحالات ارتفعت. وجدت الشبكة أن حسابات موالية بالدولة نشرت ٤٠ ألف تغريدة. ومن المرجح أن الحملة استمرت. لم نتمكن من التحقق من المصدر."
T = 0
def t(name, cond, out):
    global T; T += 1
    assert cond, (name, out)
    print("ok", T, "-", name, "->", out.get("decision"), out.get("reason"))
# 1 outlet finding binds
t("outlet-finding", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","spans":{"claim":"حسابات موالية بالدولة نشرت ٤٠ ألف تغريدة"}}', doc, 60, 120)["decision"] == "bind", out if False else h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","spans":{"claim":"حسابات موالية بالدولة نشرت ٤٠ ألف تغريدة"}}', doc, 60, 120))
# 2 document-attributed binds with claim+attribution spans (v1.3: claim required)
t("document-attributed", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"document","spans":{"claim":"ومن المرجح أن الحملة استمرت","attribution":"وزارة الصحة"}}', doc, 60, 120)["decision"] == "bind", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"document","spans":{"claim":"ومن المرجح أن الحملة استمرت","attribution":"وزارة الصحة"}}', doc, 60, 120))
# 3 actor-bloc attribution (unnamed bloc) binds with claim+verbatim attribution spans (v1.3: claim required)
t("actor-bloc-attribution", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"quoted_actor","spans":{"claim":"ومن المرجح أن الحملة استمرت","attribution":"حسابات موالية بالدولة"}}', doc, 60, 120)["decision"] == "bind", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"quoted_actor","spans":{"claim":"ومن المرجح أن الحملة استمرت","attribution":"حسابات موالية بالدولة"}}', doc, 60, 120))
# 4 hedged (calibrated) finding stays eligible - grader uncertainty absent
t("hedged-finding-eligible", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","spans":{"claim":"ومن المرجح أن الحملة استمرت"}}', doc, 60, 120)["decision"] == "bind", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","spans":{"claim":"ومن المرجح أن الحملة استمرت"}}', doc, 60, 120))
# 5a caption_analysis with quoted-source subject -> DENY (B2: no subject bypass)
t("caption-analysis-quoted-denied", h.grade_unit('{"role":["caption_analysis"],"subject_kind":"quoted_source","asserted_by":"outlet_analysis"}', doc, 60, 120)["reason"] == "subject:quoted_source", h.grade_unit('{"role":["caption_analysis"],"subject_kind":"quoted_source","asserted_by":"outlet_analysis"}', doc, 60, 120))
# 5b caption_analysis with analysis_subject -> eligible
t("caption-analysis-subject-eligible", h.grade_unit('{"role":["caption_analysis"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","spans":{"claim":"ومن المرجح أن الحملة استمرت"}}', doc, 60, 120)["decision"] == "bind", h.grade_unit('{"role":["caption_analysis"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","spans":{"claim":"ومن المرجح أن الحملة استمرت"}}', doc, 60, 120))
# 5c contradiction -> abstain like grader uncertainty (B1)
t("contradiction-abstains", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","contradiction":true}', doc, 60, 120)["reason"] == "contradiction_flagged", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","contradiction":true}', doc, 60, 120))
# 6 finding + conditional tail: finding scope survives, note recorded
o6 = h.grade_unit('{"role":["finding","conditional"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","spans":{"claim":"ومن المرجح أن الحملة استمرت"}}', doc, 60, 120)
t("finding-with-conditional-tail", o6["decision"] == "bind" and "conditional_tail_scoped_out" in o6.get("scope_notes", []), o6)
# 7 denial-only -> deny via taxonomy
t("denial-only", h.grade_unit('{"role":["denial"],"subject_kind":"quoted_source","asserted_by":"quoted_actor","spans":{"negation":"لم نتمكن من التحقق"}}', doc, 60, 120)["reason"] == "role:denial", h.grade_unit('{"role":["denial"],"subject_kind":"quoted_source","asserted_by":"quoted_actor","spans":{"negation":"لم نتمكن من التحقق"}}', doc, 60, 120))
# 8 finding+denial WITH grounded negation -> scoped deny
t("denial-scoped", h.grade_unit('{"role":["finding","denial"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","spans":{"negation":"لم نتمكن من التحقق"}}', doc, 60, 120)["reason"] == "denial_scoped", h.grade_unit('{"role":["finding","denial"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","spans":{"negation":"لم نتمكن من التحقق"}}', doc, 60, 120))
# 9 finding+denial WITHOUT grounded negation -> conservative abstain
t("denial-ungrounded", h.grade_unit('{"role":["finding","denial"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis"}', doc, 60, 120)["reason"] == "denial_scope_ungrounded", h.grade_unit('{"role":["finding","denial"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis"}', doc, 60, 120))
# 10 span unverifiable -> invalid -> abstain
t("span-unverified", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"document","spans":{"attribution":"NO SUCH TEXT"}}', doc, 60, 120)["reason"].startswith("model_invalid:span_unverified"), h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"document","spans":{"attribution":"NO SUCH TEXT"}}', doc, 60, 120))
# 11 bad enum -> invalid
t("bad-enum", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"friend"}', doc, 60, 120)["reason"] == "model_invalid:bad_asserted_by", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"friend"}', doc, 60, 120))
# 12 analyst_unnamed -> abstain
t("analyst-unnamed", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"analyst_unnamed"}', doc, 60, 120)["reason"] == "analyst_unnamed_uncorroborated", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"analyst_unnamed"}', doc, 60, 120))
# 13 grader uncertainty -> abstain even for outlet finding
t("grader-uncertain", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","uncertainty":true}', doc, 60, 120)["reason"] == "grader_uncertain", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis","uncertainty":true}', doc, 60, 120))
# 14 truncated -> abstain
t("truncated", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis"}', doc, 60, 120, truncated=True)["reason"] == "truncated", h.grade_unit('{"role":["finding"],"subject_kind":"analysis_subject","asserted_by":"outlet_analysis"}', doc, 60, 120, truncated=True))
# 15 anchor pack + context bundle
pack = h.anchor_pack(doc)
t("anchor-pack", pack["title"].startswith("شبكة") and any("حسابات" in e for e in pack["entities"]), {"decision": "pack", "reason": str(pack["entities"][:2])})
print("SELFTEST PASS:", T, "contract-required contrasts")
