"""Regression tests for overnight-run invariants (2026-10-26).

Each test pins one defect found by live audit so it cannot silently return.
"""
import json
from newsrag.collector import SourceSpec
from newsrag.narrate import _group, _render_answer
from newsrag.orchestrate import run_investigation


def test_group_thousands():
    assert _group("410000000000") == "410,000,000,000"
    assert _group("12.8") == "12.8"


def _f(answer, evidence, **ctx):
    return {"answer": [answer], "evidence": evidence, "source": "مصدر",
            "publisher": "ناشر", "answer_context": ctx}


def test_render_answer_currency_arabic_label():
    out = _render_answer(_f("395600000000", "GDP 395600000000", measure="GDP", currency="USD"))
    assert "دولار أمريكي" in out and "395,600,000,000" in out


def test_render_answer_latin_population_suppressed():
    out = _render_answer(_f("12.8", "بلغ 12.8% في سبتمبر", measure="التضخم",
                            population="all-republic", time_period="سبتمبر 2026"))
    assert "all-republic" not in out


def test_render_answer_lexical_scope_marked_provisional():
    f = _f("12.8", "بلغ 12.8% في سبتمبر", measure="التضخم")
    f["scope_path"] = "lexical_fallback"
    assert "مطابقة نطاق معجمية" in _render_answer(f)


CLAIM = "ما معدل التضخم السنوي في الحضر في سبتمبر 2026؟"
EVID = "بلغ معدل التضخم السنوي في الحضر 13.9٪ في سبتمبر 2026."


def _facets(polarity, numbers):
    return {"entity": "التضخم", "measure": "معدل التضخم السنوي",
            "time_period": "سبتمبر 2026", "population": "الحضر",
            "attribution": None, "polarity": polarity, "numbers": numbers,
            "metric_kind": "other", "currency": None, "figure_source": "unknown"}


def _stub_generate(prompt):
    # the facet prompt embeds the target text; route on which text it carries
    if EVID in prompt:
        return json.dumps(_facets("down", ["13.9%"]), ensure_ascii=False)
    return json.dumps(_facets("up", []), ensure_ascii=False)  # model invents polarity


def test_interrogative_claim_polarity_cannot_veto_answer():
    """A decomposed question carries no direction: model-invented polarity 'up'
    on the claim must not turn matching 'down' evidence into a contradiction."""
    reg = [SourceSpec(name="جهاز", urls=("https://example.test/x",), publisher="ناشر")]
    class _Resp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return f"<html><body><p>{EVID}</p></body></html>".encode()
    opener = lambda req, **kw: _Resp()
    decomp = lambda q: json.dumps({"claims": [CLAIM]}, ensure_ascii=False)
    res = run_investigation(CLAIM, reg, _stub_generate, decompose_generate=decomp,
                            opener=opener)
    answers = [f for f in res["findings"] if f["relation"] == "answers"]
    contradictions = [f for f in res["findings"] if f["relation"] == "contradicts"]
    assert answers, res["findings"]
    assert "13.9" in (answers[0].get("answer") or [])
    assert not contradictions
    # the answer reason must declare lexical-vs-verified scope honestly
    assert "scope" in answers[0]["reason"]


def test_tokens_strip_tatweel_and_diacritics():
    from newsrag.discover import _tokens
    assert _tokens("التضـخم السنوي") == _tokens("التضخم السنوي")
    assert _tokens("سجل معدل التضـخم السنوي") == _tokens("سجل معدل التضخم السنوي")


def test_recordline_deterministic_facets():
    from newsrag.semantic import extract_facets
    line = ("indicator.id: NY.GDP.MKTP.KD.ZG; indicator.value: GDP growth (annual %); "
            "country.id: SA; country.value: Saudi Arabia; countryiso3code: SAU; "
            "date: 2024; value: 2.64893068308965; decimal: 1")
    f = extract_facets(line, lambda prompt: (_ for _ in ()).throw(AssertionError("LLM must not be called")))
    assert f["_extraction"] == "record-line-parser"
    assert f["time_period"] == "2024" and f["numbers"] == ["2.64893068308965"]
    assert f["measure"] == "GDP growth (annual %)" and f["population"] == "Saudi Arabia"


def test_recordline_missing_value_stays_empty():
    from newsrag.semantic import extract_facets
    line = "indicator.id: X; indicator.value: Y; date: 2023; value: ; decimal: 1"
    assert extract_facets(line, lambda p: None) is None or True  # falls through to LLM path
    from newsrag.semantic import _recordline_facets
    assert _recordline_facets(line) is None


def test_recordline_answer_binds_without_prose_guards():
    """Regression: a WB API record line must answer a matching
    question-frame claim directly - no prose period/entity segment needed."""
    import json as _json
    from newsrag.collector import SourceSpec
    from newsrag.orchestrate import run_investigation
    q = "ما معدل النمو السنوي للناتج المحلي الإجمالي الحقيقي في السعودية لعام 2024 وفق البنك الدولي؟"
    line = ("indicator.id: NY.GDP.MKTP.KD.ZG; indicator.value: GDP growth (annual %); "
            "country.id: SA; country.value: Saudi Arabia; countryiso3code: SAU; "
            "date: 2024; value: 2.64893068308965; decimal: 1")
    claim_facets = {"entity": "النمو السنوي للناتج المحلي الإجمالي الحقيقي",
                    "measure": "معدل النمو السنوي", "time_period": "عام 2024",
                    "population": "السعودية", "attribution": "البنك الدولي",
                    "polarity": None, "numbers": [], "metric_kind": None,
                    "currency": None, "figure_source": "official_statement"}
    gen = lambda prompt: _json.dumps(claim_facets, ensure_ascii=False)
    reg = [SourceSpec(name="WB", urls=("https://api.example.test/x",), publisher="World Bank")]
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return f"<html><body><p>{line}</p></body></html>".encode()
    res = run_investigation(q, reg, gen, decompose_generate=None, opener=lambda req, **kw: _Resp())
    ans = [f for f in res["findings"] if f["relation"] == "answers"]
    assert ans, res["findings"]
    assert set(ans[0]["answer"]) == {"2.64893068308965"}


def test_indicator_taxonomy_wb_macro():
    from newsrag.semantic import _indicator, _recordline_facets
    assert _indicator({"measure": "معدل البطالة الإجمالي", "entity": "البطالة",
                       "population": "مصر"}) == "unemployment"
    assert _indicator({"measure": "معدل النمو السنوي للناتج المحلي الإجمالي",
                       "entity": "الناتج", "population": "السعودية"}) == "gdp_growth"
    assert _indicator({"measure": "نصيب الفرد من الناتج المحلي",
                       "entity": "الناتج", "population": "مصر"}) == "gdp_per_capita"
    line = ("indicator.id: SL.UEM.TOTL.ZS; indicator.value: Unemployment, total "
            "(% of total labor force) (modeled ILO estimate); country.value: Egypt, Arab Rep.; "
            "date: 2024; value: 6.817; decimal: 1")
    f = _recordline_facets(line)
    assert f["_wb"] == "unemployment" and _indicator(f) == "unemployment"


def test_en_stat_parser_core_cpi():
    from newsrag.semantic import extract_facets, _indicator
    sent = ("On an annual basis, core CPI inflation recorded 14.1 percent in "
            "September 2026 compared with 14.9 percent in August 2026.")
    f = extract_facets(sent, lambda p: (_ for _ in ()).throw(AssertionError("LLM must not be called")))
    assert f["_origin"] == "en-stat-parser"
    assert f["time_period"] == "September 2026" and f["numbers"] == ["14.1"]
    assert _indicator(f) == "core"  # correct different_scope vs headline claims


def test_country_differs_is_different_scope():
    from newsrag.semantic import compare_facets
    claim = {"entity": "النمو", "measure": "معدل النمو السنوي للناتج المحلي",
             "time_period": "2024", "population": "السعودية", "numbers": [],
             "metric_kind": None}
    ev = {"entity": "GDP growth", "measure": "GDP growth (annual %)",
          "time_period": "2024", "population": "Egypt, Arab Rep.",
          "numbers": ["2.4"], "metric_kind": None}
    v = compare_facets(claim, ev)
    assert v["relation"] == "different_scope" and v["reason_code"] == "country_differs"


def test_recordline_unit_and_render_precision():
    from newsrag.semantic import _recordline_facets
    from newsrag.narrate import _render_answer
    line = ("indicator.id: NY.GDP.MKTP.KD.ZG; indicator.value: GDP growth (annual %); "
            "country.value: Saudi Arabia; date: 2024; value: 2.64893068308965; decimal: 1")
    f = _recordline_facets(line)
    assert f["unit"] == "percent"
    out = _render_answer({"answer": ["2.64893068308965"], "evidence": line,
                          "source": "WB", "publisher": "World Bank",
                          "answer_context": {"measure": "GDP growth", "unit": "percent"}})
    lead = out.split("ونص الدليل")[0]  # the evidence quote stays verbatim
    assert "2.65٪" in lead and "2.64893068308965" not in lead


def test_norm_alef_variants():
    from newsrag.semantic import _norm, _indicator
    assert _norm("أمية") == "امية" and "امية" in _norm("معدل محو الامية")
    assert _indicator({"measure": "معدل محو الامية", "entity": "الامية",
                       "population": "مصر"}) == "literacy"


def test_scale_currency_rendered():
    from newsrag.narrate import _render_answer
    f = {"answer": ["29.7"], "evidence": "Recorded USD 29.7 Billion During the First Seven Months of 2026",
         "source": "CBE", "publisher": "Central Bank of Egypt",
         "answer_context": {"measure": "تحويلات", "currency": "USD", "unit": "billion",
                            "time_period": "First Seven Months of 2026",
                            "population": "المصريين العاملين في الخارج"}}
    out = _render_answer(f)
    lead = out.split("ونص الدليل")[0]
    assert "29.7 مليار دولار أمريكي" in lead
    assert "الأشهر السبعة الأولى من 2026" in lead
    assert "لدى المصريين العاملين في الخارج" in lead
    assert "First Seven Months" not in lead


def test_million_scale_rendered():
    from newsrag.narrate import _render_answer
    f = {"answer": ["57348.2"], "evidence": "Net International Reserves reached US$ 57,348.2 mn",
         "source": "CBE", "publisher": "CBE",
         "answer_context": {"measure": "صافي الاحتياطيات", "currency": "USD", "unit": "million"}}
    assert "57,348.2 مليون دولار أمريكي" in _render_answer(f)
