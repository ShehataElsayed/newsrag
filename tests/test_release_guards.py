"""Targeted regressions for the review-driven guards (
new guards without new tests are a coverage gap). Pure-function tests, no LLM."""
import sys

sys.path.insert(0, 'src')

from newsrag.collector import html_to_text2
from newsrag.narrate import _period_ar, _render_answer
from newsrag.orchestrate import (
    _THRESHOLD,
    _drop_delta_figures,
    dedup_answer_findings,
    prefer_measure,
)
from newsrag.semantic import _en_estimated_facets


def test_threshold_regex_marks_bounds_not_estimates():
    assert _THRESHOLD.search("Total population exceeds 35 million by mid-2024")
    assert _THRESHOLD.search("عدد السكان تجاوز 35 مليون")
    assert not _THRESHOLD.search("The total population is estimated at 35.3 million people")


def test_delta_figures_dropped_level_kept():
    sent = ("unemployment rate among Saudis reached 6.5% in Q2 of 2026, "
            "increasing by 0.1 percentage points compared to the previous quarter")
    assert _drop_delta_figures(sent, ["0.1", "6.5"]) == ["6.5"]
    # a figure not in delta position survives
    assert _drop_delta_figures("rate reached 6.5% in Q2 of 2026", ["6.5"]) == ["6.5"]


def test_prefer_measure_blocks_invented_granularity():
    # evidence-side Arabic measure adds شهري the claim never said -> claim wins
    assert prefer_measure("معدل البطالة الشهري", "معدل البطالة",
                          "ما معدل البطالة في الربع الثاني؟") == "معدل البطالة"
    # qualifier present in the claim -> evidence measure is legitimate
    assert prefer_measure("معدل البطالة الشهري", "معدل البطالة الشهري", "x") == "معدل البطالة الشهري"
    # latin evidence fragment -> claim side
    assert prefer_measure("unemployment rate", "معدل البطالة", "x") == "معدل البطالة"


def test_dedup_answer_findings_marks_second_copy():
    fs = [
        {"relation": "answers", "claim": "c1", "source": "S", "answer": ["57,348.2"]},
        {"relation": "answers", "claim": "c1", "source": "S", "answer": ["57348.2"]},
        {"relation": "answers", "claim": "c1", "source": "S", "answer": ["12.0"]},
    ]
    dedup_answer_findings(fs)
    assert fs[0]["relation"] == "answers"
    assert fs[1]["relation"] == "duplicate" and fs[1]["reason_code"] == "duplicate_answer"
    assert fs[2]["relation"] == "answers"  # different value: not a duplicate


def test_en_estimated_parser():
    f = _en_estimated_facets("The total population of the Kingdom is estimated at "
                             "35.3 million people in mid -2024, with an annual growth rate of 4.7%")
    assert f["numbers"] == ["35.3"] and f["unit"] == "million"
    assert f["time_period"] == "mid-2024"
    assert _en_estimated_facets("unrelated prose without the shape") is None


def test_scale_render_no_currency():
    base = {"source": "S", "evidence": "estimated at 35.3 million people in mid -2024",
            "answer_context": {"measure": "عدد السكان", "population": "السعودية",
                               "time_period": "mid-2024", "unit": "million", "currency": None}}
    out = _render_answer({**base, "answer": ["35.3"]})
    assert "35.3 مليون" in out and "منتصف 2024" in out
    # already-expanded figure must NOT get the scale word twice
    out2 = _render_answer({**base, "answer": ["35000000"],
                           "evidence": "Total population exceeds 35 million"})
    assert "35,000,000 مليون" not in out2 and "35,000,000" in out2


def test_period_ar_mid_year():
    assert _period_ar("mid-2024") == "منتصف 2024"


def test_meta_fallback_labels_js_shell():
    shell = (b'<html><head><meta name="description" content="SPA: census puts population '
             b'at 32,175,224."></head><body><div id="app"></div></body></html>')
    text, used = html_to_text2(shell)
    assert used and "32,175,224" in text
    rich_body = b"<html><body><p>" + b"real article body " * 60 + b"</p></body></html>"
    text2, used2 = html_to_text2(rich_body)
    assert not used2 and "real article body" in text2


def test_year_local_binding_drops_other_periods():
    from newsrag.orchestrate import _year_local
    sent = ("فقد نما عدد السكان الإجمالي من 24 مليون نسمة عام 2010م إلى "
            "32.2 مليون نسمة عام 2022م")
    cf = {"time_period": "2022"}
    keep = _year_local(sent, ["24000000", "32200000"], cf)
    assert keep == ["32200000"]
    # exact-count surface form near the census year survives
    sent2 = "the 2022 census puts the population at 32,175,224; the 2010 census had 24 million"
    assert _year_local(sent2, ["32175224", "24000000"], cf) == ["32175224"]
    # no competing year -> everything kept
    assert _year_local("بلغ 32.2 مليون عام 2022م", ["32200000"], cf) == ["32200000"]
    # claim without a year -> no filtering
    assert _year_local(sent, ["24000000"], {"time_period": None}) == ["24000000"]


def test_percent_dropped_for_count_claims():
    from newsrag.orchestrate import _drop_percent_for_nonrate
    sent = "بلغ 32.2 مليون نسمة عام 2022م بمتوسط معدل نمو سنوي وصل إلى 2.5%"
    cf = {"measure": "العدد الإجمالي للسكان"}
    assert _drop_percent_for_nonrate(sent, ["32200000", "2.5"], cf) == ["32200000"]
    # rate claim keeps its % figure
    cf2 = {"measure": "معدل النمو"}
    assert _drop_percent_for_nonrate(sent, ["2.5"], cf2) == ["2.5"]


def test_subset_figures_dropped_for_whole_scope_claims():
    from newsrag.orchestrate import _drop_subset_figures
    sent = ("Population estimates indicate that the population of the State of "
            "Palestine reached approximately 5.56 million by the end of 2025, "
            "with 3.43 million in the West Bank.")
    cf = {"population": "دولة فلسطين", "entity": "السكان", "_sentence": "ما تقديرات عدد سكان دولة فلسطين؟"}
    assert _drop_subset_figures(sent, ["5560000", "3430000"], cf) == ["5560000"]


def test_axis_paraphrase_dedup():
    from newsrag.orchestrate import dedup_paraphrase_claims
    claims = ["ما تقديرات عدد سكان دولة فلسطين في نهاية عام 2025؟",
              "ما تقديرات عدد سكان دولة فلسطين في عام 2025؟",
              "ما تقديرات عدد سكان دولة فلسطين في نهاية السنة 2025؟",
              "ما تقديرات عدد سكان دولة فلسطين في نهاية العام 2025؟"]
    assert len(dedup_paraphrase_claims(claims)) == 1
    two = ["ما معدل التضخم الحضري السنوي في سبتمبر 2026؟",
           "ما معدل التضخم لإجمالي الجمهورية في سبتمبر 2026؟"]
    assert len(dedup_paraphrase_claims(two)) == 2


def test_axis_paraphrase_dedup_keeps_distinct_measures():
    from newsrag.orchestrate import dedup_paraphrase_claims
    total = "ما هو العدد الإجمالي لسكان السعودية في 2022؟"
    saudis = "ما هو عدد السعوديين من سكان السعودية في 2022؟"
    assert dedup_paraphrase_claims([total, saudis]) == [total, saudis]


def test_subgroup_mismatch_guard_splits_saudis_total():
    from newsrag.orchestrate import _drop_subgroup_mismatch
    sent = ("Of the total number, Saudis make up 18.8 million (58.4%), "
            "while non-Saudis, 13.4 million, constitute 41.6% of the population.")
    saudis_claim = {"population": "السعوديين", "entity": "السعوديون",
                    "_sentence": "ما هو عدد السعوديين من سكان السعودية في 2022؟"}
    total_claim = {"population": "السعودية", "entity": "سكان السعودية",
                   "_sentence": "ما هو العدد الإجمالي لسكان السعودية في 2022؟"}
    figs = ["18800000", "13400000"]
    assert _drop_subgroup_mismatch(sent, figs, saudis_claim) == ["18800000"]
    assert _drop_subgroup_mismatch(sent, figs, total_claim) == []


def test_subgroup_mismatch_guard_headline_binds_total_only():
    from newsrag.orchestrate import _drop_subgroup_mismatch
    sent = ("Saudi Arabia Census Shows Total Population of 32.2 Million, "
            "of Which 18.8 Million are Saudis")
    saudis_claim = {"population": "السعوديين", "entity": "السعوديون",
                    "_sentence": "ما هو عدد السعوديين من سكان السعودية في 2022؟"}
    total_claim = {"population": "السعودية", "entity": "سكان السعودية",
                   "_sentence": "ما هو العدد الإجمالي لسكان السعودية في 2022؟"}
    figs = ["32200000", "18800000"]
    assert _drop_subgroup_mismatch(sent, figs, total_claim) == ["32200000"]
    assert _drop_subgroup_mismatch(sent, figs, saudis_claim) == ["18800000"]


def test_rescale_bound_applies_sentence_scale():
    from newsrag.orchestrate import _rescale_bound
    sent = "The total population of the Kingdom is estimated at 35.3 million people in mid-2024."
    assert _rescale_bound(sent, ["35.3"]) == ["35300000"]


def test_document_anchor_years_requires_single_substantive_year():
    from newsrag.orchestrate import _document_anchor_years
    # announcement date is metadata: single substantive anchor survives
    assert _document_anchor_years("Census 2022 (results announced 2023-05-31)") == ["2022"]
    # multi-period document: no anchor, a claim-asked year must NOT be inherited
    assert _document_anchor_years("2010 إلى 2022") == []
    assert _document_anchor_years("2011-2024 time series") == []
    # no year at all: nothing to inherit
    assert _document_anchor_years("mid-year estimates") == []
    assert _document_anchor_years("September 2026") == ["2026"]


def test_extraction_generate_error_negative_cached():
    """A generation-side failure (timeout/transport) is classified and
    negative-cached with its reason: a slow candidate is not re-attempted at
    full cost on every resume, and the failure reason stays inspectable."""
    import importlib

    import newsrag.semantic as sem
    importlib.reload(sem)
    seen = {"calls": 0}

    def boom(prompt):
        seen["calls"] += 1
        raise TimeoutError("budget")

    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    os.environ["NEWSRAG_EXCACHE"] = path
    try:
        assert sem.extract_facets("unique text for timeout guard 123", boom) is None
        assert sem.extract_facets("unique text for timeout guard 123", boom) is None
        assert seen["calls"] == 1  # second call served from negative cache
        import json
        with open(path) as fh:
            entry = json.load(fh)[sem._cache_key("unique text for timeout guard 123")]
        assert entry["_failed"] and "TimeoutError" in entry["_reason"]
    finally:
        del os.environ["NEWSRAG_EXCACHE"]
        os.unlink(path)
        importlib.reload(sem)


def test_decompose_cache_key_invalidation():
    """Decompose cache key binds question + model + prompt + grammar + version:
    changing any one invalidates old entries (no stale decomposition replay),
    and hit/fresh accounting is recorded."""
    import importlib
    import os
    import tempfile

    import newsrag.semantic as sem
    importlib.reload(sem)
    q = "ما معدل التضخم في سبتمبر 2026؟"
    k0 = sem._decomp_cache_key(q)
    assert k0 == sem._decomp_cache_key(q)  # deterministic
    assert k0 != sem._decomp_cache_key(q + " مختلف")  # question change
    os.environ["NEWSRAG_MODEL_ID"] = "other-model.gguf"
    try:
        assert sem._decomp_cache_key(q) != k0  # model change invalidates
    finally:
        del os.environ["NEWSRAG_MODEL_ID"]
    with tempfile.NamedTemporaryFile(suffix=".gbnf", delete=False) as f:
        f.write(b"altered grammar"); gp = f.name
    os.environ["NEWSRAG_CLAIMS_GRAMMAR"] = gp
    try:
        assert sem._decomp_cache_key(q) != k0  # grammar change invalidates
    finally:
        del os.environ["NEWSRAG_CLAIMS_GRAMMAR"]; os.unlink(gp)
    v = sem._DECOMP_VERSION
    sem._DECOMP_VERSION = v + "-bump"
    try:
        assert sem._decomp_cache_key(q) != k0  # version bump invalidates
    finally:
        sem._DECOMP_VERSION = v
    importlib.reload(sem)


def test_decompose_hit_fresh_accounting():
    """decompose_question records fresh vs hit counts and serves the second
    identical call from cache without re-invoking the model."""
    import importlib
    import json
    import os
    import tempfile

    import newsrag.semantic as sem
    importlib.reload(sem)
    import newsrag.orchestrate as orch
    importlib.reload(orch)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    os.environ["NEWSRAG_EXCACHE"] = path
    calls = {"n": 0}
    def fake_gen(prompt):
        calls["n"] += 1
        return json.dumps({"claims": ["ادعاء واحد 2026"]}, ensure_ascii=False)
    try:
        q = "سؤال محاسبي فريد 2026؟"
        a = orch.decompose_question(q, fake_gen)
        b = orch.decompose_question(q, fake_gen)
        assert a == b and calls["n"] == 1
        assert sem._DECOMP_STATS["fresh"] >= 1 and sem._DECOMP_STATS["hits"] >= 1
    finally:
        del os.environ["NEWSRAG_EXCACHE"]; os.unlink(path)
        importlib.reload(sem); importlib.reload(orch)


def test_coordination_bar_caption_not_analysis():
    """A coordination claim (حملات منسقة) requires analysis-level coordination
    evidence: a photo caption repeating the slogan, a heading fragment, or a
    shared-sentiment inference are recorded but NOT bound as answers (caption/analysis/inference distinction)."""
    from newsrag.orchestrate import (
        _coordination_grade,
        _coordination_required,
        _is_caption_or_heading,
    )
    assert _coordination_required("هل وثّقت تحقيقات حملات منسقة ضد المصريين؟")
    assert _coordination_required("Were coordinated campaigns documented?")
    assert not _coordination_required("ما معدل التضخم في سبتمبر 2026؟")
    # captions/headings are not analysis
    assert _is_caption_or_heading("صورة تداولتها حسابات إلكترونية سعودية ضمن حملة رقمية منسقة للتحريض")
    assert _is_caption_or_heading("Unified Discourse and Cohesive Network")
    assert _is_caption_or_heading("24, 2026 Fake Accounts, Thousands of Posts")
    # inference/sentiment is not coordination analysis
    assert not _coordination_grade("These figures suggest a regional alignment or shared sentiment with a dominant Saudi discourse against Egyptians online")
    # analysis-level coordination sentences pass the bar
    assert _coordination_grade("This may indicate the presence of centralized coordination through a network of accounts amplifying the campaign")
    assert _coordination_grade("This model clearly indicates a well-coordinated network of accounts driving the posts")
    assert _coordination_grade("تحليل البيانات يكشف عن شبكة منسقة من الحسابات الوهمية")


def test_summary_counts_provisional_binding_as_partial_not_settled():
    """A provisional lexical-scope binding documents an axis partially; the
    cross-axis summary must not claim it as settled with documented evidence."""
    from newsrag.narrate import assemble_narrative
    result = {
        "decomposed_claims": [{"claim": "محور أول"}, {"claim": "محور ثان"}],
        "findings": [
            {"claim": "محور أول", "relation": "answers", "source": "S1",
             "evidence": "e", "answer": ["242"], "scope_path": "lexical_fallback"},
        ],
        "gaps": [], "provenance": [],
    }
    text = assemble_narrative("سؤال", result)
    assert "سُوِّي 0 من 2 محاور بأدلة موثقة" in text
    assert "توثيقاً جزئياً" in text
    # and a verified-scope binding settles the axis again
    result["findings"][0]["scope_path"] = "indicator"
    text2 = assemble_narrative("سؤال", result)
    assert "سُوِّي 1 من 2 محاور بأدلة موثقة" in text2
    assert "توثيقاً جزئياً" not in text2


def test_coordination_claim_carries_coordination_indicator():
    """Root-cause fix for the review: a coordination-required claim must
    carry a 'coordination' indicator facet, not collapse to None. Volume-metric
    evidence is then honestly different-scope; analysis-level coordination
    evidence is the only same-scope match (the orchestrate gate still decides
    whether it binds)."""
    from newsrag.semantic import _indicator, compare_facets
    claim = {"entity": "حملات منسقة ضد المصريين", "measure": None,
             "population": "المصريون في السعودية",
             "_sentence": "هل وثّقت تحقيقات صحفية حملات منسقة ضد المصريين في السعودية خلال عام 2025؟"}
    assert _indicator(claim) == "coordination"
    volume = {"entity": "campaign posts", "measure": "posts", "population": "United Kingdom",
              "metric_kind": "volume",
              "_sentence": "Participation extended to the United Kingdom (242 posts)."}
    assert _indicator(volume) == "campaign_volume"
    r = compare_facets(dict(claim), dict(volume))
    assert r["relation"] == "different_scope" and r["reason_code"] == "indicator_differs"
    assert "claim=coordination" in r["reason"]
    analysis = {"entity": "شبكة منسقة من الحسابات الوهمية", "measure": None, "population": None,
                "_sentence": "تحليل البيانات يكشف عن شبكة منسقة من الحسابات الوهمية"}
    assert _indicator(analysis) == "coordination"
    r2 = compare_facets(dict(claim), dict(analysis))
    assert r2.get("reason_code") != "indicator_differs"


def test_narrative_keeps_distinct_qualitative_answers_from_same_source():
    """Qualitative bindings carry no answer figures; the dedup key must not
    collapse distinct evidence sentences from one source into one line."""
    from newsrag.narrate import assemble_narrative
    result = {
        "decomposed_claims": [{"claim": "حملات منسقة"}],
        "findings": [
            {"claim": "حملات منسقة", "relation": "answers", "source": "S1",
             "evidence": "well-coordinated and carefully planned digital campaign", "answer": []},
            {"claim": "حملات منسقة", "relation": "answers", "source": "S1",
             "evidence": "centralized coordination to create new, synchronized posts", "answer": []},
        ],
        "gaps": [], "provenance": [],
    }
    text = assemble_narrative("سؤال", result)
    assert "well-coordinated and carefully planned" in text
    assert "centralized coordination" in text


def test_heading_methodology_not_coordination_grade():
    """Review: 'Indicators of a Coordinated Campaign Using Meltwater, the
    data collection process began...' is a heading + methodology sentence, not
    an analysis conclusion; the two hedged analysis findings stay eligible."""
    from newsrag.orchestrate import _coordination_grade
    assert not _coordination_grade(
        "Indicators of a Coordinated Campaign Using Meltwater, the data collection "
        "process began by examining and analyzing the daily activity of posts")
    assert _coordination_grade(
        "This model clearly indicates a well-coordinated and carefully planned "
        "digital campaign, rather than an organic or natural interaction between users.")
    assert _coordination_grade(
        "This may indicate the presence of centralized coordination to create new, "
        "synchronized posts with a unified message.")


def test_decompose_drops_subset_paraphrase_claim():
    from newsrag.orchestrate import _dedup_paraphrase_claims
    out = _dedup_paraphrase_claims([
        "تحقيقات صحفية وثّقت حملات منسقة ضد المصريين في السعودية خلال عام 2025",
        "تحقيقات صحفية وثّقت حملات منسقة ضد المصريين في السعودية",
    ])
    assert out == ["تحقيقات صحفية وثّقت حملات منسقة ضد المصريين في السعودية خلال عام 2025"]


def test_single_source_settlement_is_qualified():
    from newsrag.narrate import assemble_narrative
    result = {
        "decomposed_claims": [{"claim": "أ"}, {"claim": "ب مختلف تماماً"}],
        "findings": [
            {"claim": "أ", "relation": "answers", "source": "S1", "evidence": "e1",
             "answer": ["1"], "scope_path": "indicator"},
        ],
        "gaps": [], "provenance": [],
    }
    text = assemble_narrative("سؤال", result)
    assert "وفق تحليل مصدر واحد" in text
    result["findings"].append({"claim": "أ", "relation": "answers", "source": "S2",
                               "evidence": "e2", "answer": ["2"], "scope_path": "indicator"})
    assert "وفق تحليل مصدر واحد" not in assemble_narrative("سؤال", result)


def test_dedup_paraphrase_edge_cases():
    from newsrag.orchestrate import _dedup_paraphrase_claims
    # non-subset siblings both survive
    out = _dedup_paraphrase_claims(["تضخم الحضر في سبتمبر 2026", "معدل البطالة في الربع الثاني 2026"])
    assert len(out) == 2
    # identical claims collapse to one
    out2 = _dedup_paraphrase_claims(["حملات منسقة 2025", "حملات منسقة 2025"])
    assert out2 == ["حملات منسقة 2025"]
    # empty input stays empty; singleton untouched
    assert _dedup_paraphrase_claims([]) == []
    assert _dedup_paraphrase_claims(["سؤال واحد"]) == ["سؤال واحد"]


def test_coordination_grade_arabic_hedged_and_english_suggest():
    from newsrag.orchestrate import _coordination_grade
    assert _coordination_grade("يرصد التحليل وجود شبكة منسقة من الحسابات الوهمية تدير الحملة")
    assert _coordination_grade("The data suggests centralized coordination behind the synchronized posts.")
    # verb-less leading title in Arabic is not analysis either
    assert not _coordination_grade("مؤشرات حملة منسقة، بدأت عملية جمع البيانات بفحص المنشورات")
    # sentiment/regional alignment inference still excluded
    assert not _coordination_grade("The shared sentiment and regional alignment suggest a common cause.")


def test_coordination_indicator_english_claim_and_orchestrated_evidence():
    from newsrag.semantic import _indicator, compare_facets
    claim = {"entity": "coordinated campaigns against Egyptians", "measure": None,
             "population": "Egyptians in Saudi Arabia",
             "_sentence": "Did investigations document coordinated campaigns against Egyptians in Saudi Arabia in 2025?"}
    assert _indicator(claim) == "coordination"
    ev = {"entity": "network of fake accounts", "measure": None, "population": None,
          "_sentence": "The investigation reveals an orchestrated network of inauthentic accounts."}
    assert _indicator(ev) == "coordination"
    r = compare_facets(dict(claim), dict(ev))
    assert r.get("reason_code") != "indicator_differs"


def test_period_month_within_claim_year_is_containment():
    """Review: 'June 2025' evidence against a '2025' claim is containment,
    not a period difference. A month-specific claim with year-only evidence
    abstains; two different named months still differ."""
    from newsrag.semantic import compare_facets
    base_c = {"entity": "حملات منسقة", "measure": None, "population": "المصريين في السعودية",
              "time_period": "2025", "_sentence": "حملات منسقة ضد المصريين 2025"}
    base_e = {"entity": "حملات منسقة", "measure": None, "population": "المصريين في السعودية",
              "time_period": "June 2025", "_sentence": "حملات منسقة ضد المصريين في يونيو 2025"}
    r = compare_facets(dict(base_c), dict(base_e))
    assert r.get("reason_code") != "period_differs"
    r2 = compare_facets(dict(base_c, time_period="يونيو 2025"), dict(base_e, time_period="2025"))
    assert r2["relation"] == "insufficient" and r2["reason_code"] == "month_one_sided"
    r3 = compare_facets(dict(base_c, time_period="يونيو 2025"), dict(base_e, time_period="مايو 2025"))
    assert r3["relation"] == "different_scope" and r3["reason_code"] == "period_differs"
