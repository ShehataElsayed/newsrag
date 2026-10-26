"""Assistant pipeline tests with a scripted generator. No network, no live model."""
import json
import unittest
from datetime import datetime, timezone

from newsrag import NewsroomRAG, Source, ValidationError
from newsrag.assist import assist

SRC_A = Source(id="a", title="تقرير المناخ", text="ارتفعت درجات حرارة البحر في المنطقة خلال العقد الأخير بشكل قياسي. تؤثر موجات الحر البحرية على الثروة السمكية والنظم البيئية.", url="https://climate.example.org/report", published_at=datetime(2026, 9, 1, tzinfo=timezone.utc))
SRC_B = Source(id="b", title="تغطية صحفية", text="قال صيادون إن الصيد تراجع في السنوات الأخيرة مع ارتفاع حرارة المياه. هذه جملة إضافية طويلة عن تفاصيل أخرى لا صلة لها.", url="https://news.example.net/story", published_at=datetime(2026, 9, 2, tzinfo=timezone.utc))

DRAFT = '{"answer":"خلاصة مفصلة. ارتفعت درجات حرارة البحر في المنطقة خلال العقد الأخير. ادعاء بلا سند عن خليج السويس.","claims":["ارتفعت درجات حرارة البحر في المنطقة خلال العقد الأخير","انخفض الصيد في خليج السويس بنسبة محددة"]}'
REWRITE = '{"answer":"خلاصة منقحة: ارتفعت درجات حرارة البحر في المنطقة خلال العقد الأخير بشكل قياسي. لم نعثر على سند موثق لادعاء خليج السويس فحذف."}'
REWRITE_BAD = '{"answer":"خلاصة منقحة: ارتفعت درجات حرارة البحر. حدثت أزمة غذاء عالمية في المنطقة نفسها العام الماضي."}'
CRITIC = '{"contradiction":0.1}'
EXTRACT_OK = '{"claims":[]}'


def gen(script):
    def _g(prompt: str) -> str:
        if "Rewrite the ANSWER" in prompt:
            return script.get("rewrite", REWRITE)
        if "Extract up to 5 checkable" in prompt:
            return script.get("extract", EXTRACT_OK)
        if "skeptical reviewer" in prompt:
            return script.get("critic", CRITIC)
        return script.get("draft", DRAFT)
    return _g


def make_rag(script, with_sources=True):
    rag = NewsroomRAG(generate=gen(script))
    if with_sources:
        rag.add(SRC_A, SRC_B)
    return rag


class AssistTests(unittest.TestCase):
    def test_requires_question_and_generator(self):
        rag = NewsroomRAG(generate=gen({}))
        with self.assertRaises(ValidationError):
            assist(rag, "  ")
        with self.assertRaises(ValidationError):
            assist(NewsroomRAG(), "سؤال")

    def test_qualifies_unsupported_claim(self):
        r = assist(make_rag({}), "كيف تؤثر موجات الحر البحرية؟")
        self.assertEqual(r.decision, "answer")
        self.assertTrue(r.qualified)
        self.assertIn("منقحة", r.text)
        sup = [c for c in r.claims if c.supported]
        self.assertEqual(len(sup), 1)
        self.assertTrue(sup[0].evidence and sup[0].evidence_ref)
        self.assertLessEqual(r.score, 69)
        self.assertFalse(r.validated_for_release)

    def test_rewrite_injection_degrades_to_extractive(self):
        r = assist(make_rag({"rewrite": REWRITE_BAD, "extract": '{"claims":["حدثت أزمة غذاء عالمية في المنطقة"]}'}), "سؤال عن الحر")
        self.assertEqual(r.decision, "extractive")
        self.assertNotIn("أزمة غذاء", r.text)
        self.assertIn("Source says", r.text)

    def test_all_unsupported_is_insufficient(self):
        script = {"draft": '{"answer":"نص","claims":["ادعاء مختلق تماما عن موضوع بعيد جدا","ادعاء ثان بعيد أيضا لا علاقة له"]}'}
        r = assist(make_rag(script), "سؤال")
        self.assertEqual(r.decision, "insufficient")
        self.assertEqual(r.text, "")
        self.assertLessEqual(r.score, 39)
        self.assertEqual(r.insufficient_reason, "unsupported_claims")
        self.assertTrue(any("may exist outside this corpus" in w for w in r.warnings),
                        "never teach that abstention means absence")
        self.assertEqual(r.as_dict()["insufficient_reason"], "unsupported_claims")

    def test_no_checkable_claims_reason(self):
        script = {"draft": '{"answer":"خلاصة عامة بلا ادعاءات قابلة للفحص.","claims":[]}'}
        r = assist(make_rag(script), "سؤال")
        self.assertEqual(r.decision, "insufficient")
        self.assertEqual(r.insufficient_reason, "no_checkable_claims")

    def test_answer_has_no_reason(self):
        script = {"draft": '{"answer":"تؤثر موجات الحر البحرية على الثروة السمكية.","claims":["تؤثر موجات الحر البحرية على الثروة السمكية"]}'}
        r = assist(make_rag(script), "موجات الحر البحرية")
        self.assertEqual(r.decision, "answer")
        self.assertEqual(r.insufficient_reason, "")

    def test_markers_rederived_not_model_written(self):
        script = {"draft": '{"answer":"تؤثر موجات الحر البحرية على الثروة السمكية [E9].","claims":["تؤثر موجات الحر البحرية على الثروة السمكية"]}'}
        r = assist(make_rag(script), "موجات الحر البحرية")
        self.assertEqual(r.decision, "answer")
        self.assertNotIn("[E9]", r.text)
        self.assertIn("[E1]", r.text)

    def test_unusable_draft_fails_closed(self):
        r = assist(make_rag({"draft": "not json"}), "سؤال")
        self.assertEqual(r.decision, "insufficient")
        self.assertEqual(r.text, "")
        self.assertEqual(r.insufficient_reason, "draft_unusable")
        self.assertTrue(any("generation failure" in w for w in r.warnings))


if __name__ == "__main__":
    unittest.main()

class SalvageTests(unittest.TestCase):
    def test_truncated_draft_mid_sentence_drops_tail(self):
        # token cut before the answer string closes: tail fragment must never surface
        from newsrag.assist import _salvage_json
        raw = '{"answer":"خلاصة مفصلة. ارتفعت درجات حرارة البحر في المنطقة خلال العقد الأخير. جملة مبتورة عن خليج'
        salvaged = _salvage_json(raw)
        self.assertIsNotNone(salvaged)
        self.assertNotIn("جملة مبتورة", salvaged["answer"])
        self.assertIn("العقد الأخير.", salvaged["answer"])
        script = {"draft": raw}
        r = assist(make_rag(script), "ما أثر الحر؟")
        self.assertNotIn("جملة مبتورة", r.text)
        self.assertTrue(any("truncated" in w for w in r.warnings))

    def test_truncated_single_fragment_is_insufficient(self):
        script = {"draft": '{"answer":"خلاصة مبتورة بلا نهاية ولا نقطة'}
        r = assist(make_rag(script), "ما أثر الحر؟")
        self.assertEqual(r.decision, "insufficient")
        self.assertEqual(r.text, "")

    def test_truncated_claims_keep_only_closed_strings(self):
        script = {"draft": '{"answer":"خلاصة مفصلة. ارتفعت درجات حرارة البحر في المنطقة خلال العقد الأخير.","claims":["ادعاء كامل مغلق","ادعاء مبتور بلا نهاية'}
        r = assist(make_rag(script), "ما أثر الحر؟")
        texts = [c.text for c in r.claims]
        self.assertIn("ادعاء كامل مغلق", texts)
        self.assertNotIn("ادعاء مبتور بلا نهاية", texts)

    def test_dangling_backslash_truncation_clean(self):
        script = {"draft": '{"answer":"خلاصة مفصلة. ارتفعت درجات حرارة البحر في المنطقة خلال العقد الأخير.","claims":["ادعاء كامل مغلق"],"x":"\\'}
        r = assist(make_rag(script), "ما أثر الحر؟")
        self.assertFalse(r.text.endswith("\\"))


SRC_NUM = Source(id="n", title="بيان صحي", text="أعلنت الوزارة تطعيم 225 ألف طفل في سبتمبر. لم تعلن انتهاء الحملة بعد.", url="https://health.example.org/statement", published_at=datetime(2026, 9, 5, tzinfo=timezone.utc))

class AssistNumericGuardTests(unittest.TestCase):
    def _rag(self, claim):
        script = {"draft": '{"answer":"%s [S1]","claims":["%s"],"unknowns":[]}' % (claim, claim)}
        rag = NewsroomRAG(generate=gen(script))
        rag.add(SRC_NUM)
        return rag

    def test_false_number_claim_is_rejected(self):
        r = assist(self._rag("طعّمت الوزارة 900 ألف طفل في سبتمبر"), "كم طفلاً طعمت الوزارة؟")
        self.assertNotEqual(r.decision, "answer")
        self.assertNotIn("900", r.text)
        self.assertEqual(r.insufficient_reason, "unsupported_claims")

    def test_matching_number_claim_is_supported(self):
        r = assist(self._rag("طعّمت الوزارة 225 ألف طفل في سبتمبر"), "كم طفلاً طعمت الوزارة؟")
        self.assertEqual(r.decision, "answer")
        self.assertIn("225", r.text)

    def test_eastern_arabic_digits_match_western_source(self):
        r = assist(self._rag("طعّمت الوزارة ٢٢٥ ألف طفل في سبتمبر"), "كم طفلاً طعمت الوزارة؟")
        self.assertEqual(r.decision, "answer")

    def test_negation_parity_still_applies(self):
        neg_src = Source(id="ng", title="بيان", text="لم تعلن الوزارة انتهاء حملة التطعيم في المحافظات بعد، بحسب البيان الصادر اليوم.", url="https://health.example.org/neg", published_at=datetime(2026, 9, 5, tzinfo=timezone.utc))
        script = {"draft": '{"answer":"أعلنت الوزارة انتهاء حملة التطعيم في المحافظات [S1]","claims":["أعلنت الوزارة انتهاء حملة التطعيم في المحافظات"],"unknowns":[]}'}
        rag = NewsroomRAG(generate=gen(script))
        rag.add(neg_src)
        r = assist(rag, "هل انتهت الحملة؟")
        self.assertNotEqual(r.decision, "answer")
        self.assertEqual(r.insufficient_reason, "unsupported_claims")


class DecisionTraceTests(unittest.TestCase):
    """Explanations must match decisions; every check is auditable."""

    def _run(self, script=None, with_sources=True):
        return assist(make_rag(script or {}, with_sources), "ماذا حدث لدرجات حرارة البحر؟")

    def test_supported_answer_has_pass_trace(self):
        script = {"draft": '{"answer":"ارتفعت درجات حرارة البحر في المنطقة خلال العقد الأخير بشكل قياسي.","claims":["ارتفعت درجات حرارة البحر في المنطقة خلال العقد الأخير"]}'}
        r = self._run(script)
        self.assertEqual(r.decision, "answer")
        names = [t.name for t in r.trace]
        self.assertIn("retrieval", names)
        self.assertIn("claim_support", names)
        self.assertEqual(r.trace[-1].name, "decision")
        self.assertEqual(r.trace[-1].detail["decision"], r.decision)
        claim_steps = [t for t in r.trace if t.name == "claim_support"]
        self.assertTrue(all(t.outcome == "pass" for t in claim_steps))
        # explanation text reflects the decision and checks
        ex = r.explain()
        self.assertIn("decision: answer", ex)
        self.assertIn("claim_support", ex)

    def test_numeric_counterexample_trace_shows_missing_number(self):
        # Claim states 900; source states 225. Guard must fail with the number named.
        src = Source(id="n", title="إحصاء", text="بلغ عدد الصيادين المتضررين 225 صيادًا وفق التقرير الرسمي. هذه جملة طويلة كافية للاختبار.",
                     url="https://stats.example.org/x", published_at=datetime(2026, 9, 3, tzinfo=timezone.utc))
        rag = NewsroomRAG(generate=gen({"draft": '{"answer":"بلغ العدد 900.","claims":["بلغ عدد الصيادين المتضررين 900 صيادًا"]}'}))
        rag.add(src)
        r = assist(rag, "كم عدد الصيادين المتضررين؟")
        self.assertEqual(r.decision, "insufficient")
        self.assertEqual(r.insufficient_reason, "unsupported_claims")
        claim_steps = [t for t in r.trace if t.name == "claim_support"]
        self.assertEqual(len(claim_steps), 1)
        st = claim_steps[0]
        self.assertEqual(st.outcome, "fail")
        self.assertIn("900", st.detail["numbers_missing"])
        self.assertIn("numbers_not_in_support", st.detail["failed_checks"])
        # nearest rejected candidate is recorded for audit (E1), not accepted as support
        self.assertEqual(st.detail["evidence_ref"], "E1")
        self.assertFalse(r.claims[0].supported)
        self.assertEqual(r.claims[0].evidence_ref, "")  # nothing accepted as support
        ex = r.explain()
        self.assertIn("900", ex)
        self.assertIn("decision: insufficient", ex)

    def test_processing_failure_distinguished_from_evidence(self):
        script = {"draft": "not json at all"}
        r = self._run(script)
        self.assertEqual(r.insufficient_reason, "draft_unusable")
        kinds = {t.name: t.kind for t in r.trace}
        self.assertEqual(kinds["draft_generation"], "processing")
        self.assertEqual(kinds["decision"], "processing")

    def test_every_claim_row_has_trace_step_and_offsets(self):
        r = self._run()  # default DRAFT: one supported, one unsupported claim
        claim_steps = [t for t in r.trace if t.name == "claim_support"]
        self.assertEqual(len(claim_steps), len(r.claims))
        supported = [t for t in claim_steps if t.outcome == "pass"]
        for t in supported:
            self.assertTrue(t.detail["evidence_ref"])
            self.assertGreaterEqual(t.detail["offset"], 0)
        failed = [t for t in claim_steps if t.outcome == "fail"]
        for t in failed:
            self.assertIn("failed_checks", t.detail)

    def test_trace_serialized_in_as_dict(self):
        d = self._run().as_dict()
        self.assertIn("trace", d)
        self.assertTrue(all({"name", "outcome", "criterion", "kind"} <= set(t) for t in d["trace"]))


class OutputCoverageGuardTests(unittest.TestCase):
    """A shown verified answer must not carry unchecked prose (regression:
    live Qwen e2e run 2026-10-26 passed one truncated compound claim while
    the answer text contained multiple unchecked assertions)."""

    def _rag(self, answer, claims, src_text=None):
        script = {"draft": json.dumps({"answer": answer, "claims": claims, "unknowns": []}, ensure_ascii=False)}
        rag = NewsroomRAG(generate=gen(script))
        rag.add(SRC_NUM)
        return rag

    def test_unsupported_answer_sentences_are_removed(self):
        answer = ("طعّمت الوزارة 225 ألف طفل في سبتمبر."
                  " وتولت عشرون دولة مختلفة حكم المدينة عبر التاريخ القديم والوسيط.")
        claims = ["طعّمت الوزارة 225 ألف طفل في سبتمبر"]
        r = assist(self._rag(answer, claims), "كم طفلاً طعمت الوزارة؟")
        self.assertEqual(r.decision, "answer")
        self.assertIn("225", r.text)
        self.assertNotIn("عشرون دولة", r.text)
        self.assertTrue(any("removed" in w for w in r.warnings))
        self.assertTrue(any(t.name == "answer_sentence_support" and t.outcome == "fail" for t in r.trace))

    def test_all_sentences_unsupported_means_withheld(self):
        answer = "تأسست المدينة في العصر الحجري الحديث قبل عشرة آلاف عام تقريبًا."
        claims = ["طعّمت الوزارة 225 ألف طفل في سبتمبر"]
        r = assist(self._rag(answer, claims), "كم طفلاً طعمت الوزارة؟")
        self.assertEqual(r.decision, "insufficient")
        self.assertEqual(r.insufficient_reason, "unsupported_claims")
        self.assertEqual(r.text, "")

    def test_overlong_claim_dropped_not_truncated(self):
        long_claim = "طعّمت الوزارة 225 ألف طفل في سبتمبر " + "ثم " * 120
        claims = [long_claim, "طعّمت الوزارة 225 ألف طفل في سبتمبر"]
        r = assist(self._rag("طعّمت الوزارة 225 ألف طفل في سبتمبر.", claims), "كم طفلاً طعمت الوزارة؟")
        self.assertTrue(all(len(c.text) <= 220 for c in r.claims))
        self.assertTrue(any(t.name == "claim_filter" for t in r.trace))


class ClauseGateAdversarialTests(unittest.TestCase):
    """Adversarial coverage cases: one sentence mixing supported and
    unsupported clauses must not pass on overall sentence similarity."""

    def _rag(self, answer, claims):
        script = {"draft": json.dumps({"answer": answer, "claims": claims, "unknowns": []}, ensure_ascii=False)}
        rag = NewsroomRAG(generate=gen(script))
        rag.add(SRC_NUM)
        return rag

    def test_mixed_true_false_single_sentence_is_dropped(self):
        # One sentence: supported vaccination clause + fabricated hospital clause.
        answer = "أعلنت الوزارة تطعيم 225 ألف طفل في سبتمبر وافتتحت عشرين مستشفى جديدًا في المحافظات."
        claims = ["أعلنت الوزارة تطعيم 225 ألف طفل في سبتمبر"]
        r = assist(self._rag(answer, claims), "ماذا أعلنت الوزارة؟")
        self.assertNotIn("مستشفى", r.text)
        self.assertTrue(any(t.name == "answer_sentence_support" and t.outcome == "fail"
                            and "failed_clause" in t.detail for t in r.trace))

    def test_fully_supported_compound_sentence_is_kept(self):
        # Both clauses are verbatim fragments of the corpus sentence.
        answer = "أعلنت الوزارة تطعيم 225 ألف طفل في سبتمبر ولم تعلن انتهاء الحملة بعد."
        claims = ["أعلنت الوزارة تطعيم 225 ألف طفل في سبتمبر", "لم تعلن الوزارة انتهاء الحملة بعد"]
        r = assist(self._rag(answer, claims), "ماذا أعلنت الوزارة؟")
        if r.decision == "answer":
            self.assertIn("انتهاء الحملة", r.text)
        else:
            # fail-closed degradation is acceptable; never a false show
            self.assertEqual(r.decision, "extractive")

    def test_short_unsupported_clause_after_comma_drops_sentence(self):
        answer = "طعّمت الوزارة 225 ألف طفل في سبتمبر، وهو أكبر برنامج تطعيم في تاريخ المنطقة."
        claims = ["طعّمت الوزارة 225 ألف طفل في سبتمبر"]
        r = assist(self._rag(answer, claims), "كم طفلاً طعمت الوزارة؟")
        self.assertNotIn("أكبر برنامج", r.text)


SRC_CAIRO = Source(id="c", title="صفحة عن القاهرة", text="القاهرة عاصمة مصر وأكبر مدنها.", url="https://example.org/cairo")


class ClaimHygieneTests(unittest.TestCase):
    """Claim-list hygiene: duplicates checked once; Arabic footnote markers stripped."""

    def _rag(self, answer, claims):
        script = {"draft": json.dumps({"answer": answer, "claims": claims, "unknowns": []}, ensure_ascii=False)}
        rag = NewsroomRAG(generate=gen(script))
        rag.add(SRC_CAIRO)
        return rag

    def test_duplicate_claims_checked_once(self):
        rag = self._rag("القاهرة عاصمة مصر.",
                        ["القاهرة عاصمة مصر", "القاهرة عاصمة مصر [٤]", "القاهرة عاصمة مصر"])
        r = assist(rag, "ما عاصمة مصر؟")
        rows = [t for t in r.trace if t.name == "claim_support"]
        self.assertEqual(len(rows), 1)
        self.assertTrue(any(t.name == "claim_filter" and "dropped_duplicate" in t.detail for t in r.trace))

    def test_arabic_indic_footnote_marker_stripped(self):
        # a copied "[٤]" footnote must not poison the numbers guard or matching
        rag = self._rag("القاهرة عاصمة مصر [٤].", ["القاهرة عاصمة مصر [٤]"])
        r = assist(rag, "ما عاصمة مصر؟")
        rows = [t for t in r.trace if t.name == "claim_support"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].outcome, "pass")


def test_sentence_split_keeps_decimal_numbers_intact():
    from newsrag.assist import _sentences
    text = "سجل معدل التضخم السنوي (12.8%) لشهر سبتمبر 2026 مقابل (12.7%) عن شهر أغسطس 2026. وبلغ الرقم القياسي 294.2 نقطة."
    sents = _sentences(text)
    assert any("12.8%" in s and "12.7%" in s for s in sents), sents
    assert not any(s.strip().endswith("(12") for s in sents), sents


def test_support_accepts_claim_with_decimal_numbers_present():
    from newsrag.assist import _support_detail, _sentences_with_offsets
    corpus = "سجل معدل التضخم السنوي لإجمالي الجمهورية (12.8%) لشهر سبتمبر 2026 مقابل (12.7%) عن شهر أغسطس 2026."
    sents = [(s, "E1", off) for s, off in _sentences_with_offsets(corpus)]
    claim = "في شهر سبتمبر 2026، سجل معدل التضخم السنوي لإجمالي الجمهورية 12.8%، مقارنة بـ 12.7% في أغسطس 2026."
    ok, sc, sent, ref, det = _support_detail(claim, sents, 0.4)
    assert ok, det


def test_answer_degrades_to_extractive_when_verified_claim_dropped_from_prose():
    # Draft answers the question with a verifiable headline claim, but the
    # draft's answer text buries it in a compound sentence whose extra clause
    # fails the clause gate, while tangential supported prose survives. The
    # pipeline must degrade to extractive output conveying the verified claim,
    # not show unrelated prose as the answer.
    corpus = ("سجل معدل التضخم السنوي لإجمالي الجمهورية (12.8%) لشهر سبتمبر 2026 مقابل (12.7%) عن شهر أغسطس 2026. "
              "وارتفعت أسعار مجموعة الخضروات بنسبة (15.1%) خلال سبتمبر 2026 مقارنة بشهر أغسطس 2026.")
    src = Source(id="s1", title="bulletin", text=corpus)
    rag = NewsroomRAG(generate=None)
    rag.add(src)
    headline = "سجل معدل التضخم السنوي لإجمالي الجمهورية 12.8% في سبتمبر 2026 مقابل 12.7% في أغسطس 2026"
    tangential = "وارتفعت أسعار مجموعة الخضروات بنسبة 15.1% خلال سبتمبر 2026 مقارنة بشهر أغسطس 2026"
    # answer text: headline glued to an unsupported clause, plus tangential prose
    answer_text = headline + " وهو ما يعكس نجاح السياسة النقدية تمامًا. " + tangential
    draft = json.dumps({"answer": answer_text, "claims": [headline]}, ensure_ascii=False)
    rag.generate = lambda prompt: draft
    r = assist(rag, "ما معدل التضخم السنوي لإجمالي الجمهورية؟", top_k=3, max_claims=3, critic=False)
    assert r.decision == "extractive", (r.decision, r.text)
    assert "12.8" in r.text and "15.1" not in r.text.split("Source says")[0], r.text
    assert any(t.name == "answer_relevance" and t.outcome == "fail" for t in r.trace)
