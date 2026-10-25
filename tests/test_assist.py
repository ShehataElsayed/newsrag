"""Assistant pipeline tests with a scripted generator. No network, no live model."""
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

