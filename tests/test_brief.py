import json
import unittest
from datetime import datetime, timezone

from newsrag import NewsroomRAG, Source
from newsrag.assist import assist
from newsrag.brief import format_brief

SRC = Source(id="n", title="بيان صحي",
             text="أعلنت الوزارة تطعيم 225 ألف طفل في سبتمبر. لم تعلن انتهاء الحملة بعد.",
             url="https://health.example.org/statement",
             published_at=datetime(2026, 9, 5, tzinfo=timezone.utc))


def gen(script):
    def _g(prompt):
        if "Rewrite the ANSWER" in prompt or "Extract up to" in prompt or "skeptical" in prompt:
            return script.get("rewrite", "{}")
        return script["draft"]
    return _g


def rag_with(draft):
    r = NewsroomRAG(generate=gen({"draft": draft}))
    r.add(SRC)
    return r


class FormatBriefTests(unittest.TestCase):
    def test_supported_brief_separates_facts_and_limits(self):
        claim = "طعّمت الوزارة 225 ألف طفل في سبتمبر"
        draft = json.dumps({"answer": claim, "claims": [claim], "unknowns": []}, ensure_ascii=False)
        brief = format_brief(assist(rag_with(draft), "كم طفلاً طعمت الوزارة؟"))
        assert "الخلاصة:" in brief
        assert "ما تحقق" in brief
        assert claim in brief
        assert "حدود الفحص" in brief
        assert "ليس استلزامًا" in brief

    def test_unsupported_claim_listed_as_unverified_not_shown_as_fact(self):
        good = "طعّمت الوزارة 225 ألف طفل في سبتمبر"
        bad = "طعّمت الوزارة 900 ألف طفل في سبتمبر"
        draft = json.dumps({"answer": good, "claims": [good, bad], "unknowns": []}, ensure_ascii=False)
        r = assist(rag_with(draft), "كم طفلاً طعمت الوزارة؟", critic=False)
        brief = format_brief(r)
        assert "ما لم يتحقق" in brief
        assert "900" in brief.split("ما لم يتحقق")[1]
        # the 900 claim must not appear in the facts section
        facts = brief.split("ما تحقق")[1].split("ما لم يتحقق")[0]
        assert "900" not in facts

    def test_insufficient_brief_states_reason_and_never_promises(self):
        bad = "طعّمت الوزارة 900 ألف طفل في سبتمبر"
        draft = json.dumps({"answer": bad, "claims": [bad], "unknowns": []}, ensure_ascii=False)
        brief = format_brief(assist(rag_with(draft), "كم طفلاً طعمت الوزارة؟"))
        assert "غير كافية" in brief
        assert "بلا سند" in brief or "ادعاءات" in brief


if __name__ == "__main__":
    unittest.main()
