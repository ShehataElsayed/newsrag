"""Regression tests for the semantic comparator (see examples/adversarial/)."""
from newsrag.semantic import compare_facets


class TestComparatorRegressions:
    """Regression tests for the 2026-10-26 comparator fixes (year-aware period,
    attribution checked before direction, monthly indicator tag)."""

    def _f(self, **kw):
        base = {"entity": "التضخم", "measure": "التضخم السنوي", "time_period": None,
                "population": "الحضر", "attribution": None, "polarity": "none", "numbers": ["13.9%"]}
        base.update(kw)
        return base

    def test_same_number_wrong_year_is_different_scope_not_supports(self):
        c = self._f(time_period="سبتمبر 2025")
        s = self._f(time_period="سبتمبر 2026")
        v = compare_facets(c, s)
        assert v["relation"] == "different_scope" and v["dimension"] == "time"

    def test_year_on_one_side_only_abstains(self):
        c = self._f(time_period="سبتمبر")
        s = self._f(time_period="سبتمبر 2026")
        assert compare_facets(c, s)["relation"] == "insufficient"

    def test_attribution_mismatch_never_contradiction_even_with_opposite_direction(self):
        c = self._f(time_period="سبتمبر 2026", attribution="الجهاز المركزي للتعبئة العامة والإحصاء",
                    polarity="up", numbers=["14.5%"])
        s = self._f(time_period="سبتمبر 2026", attribution="البنك المركزي المصري",
                    polarity="down", numbers=["13.9%"])
        v = compare_facets(c, s)
        assert v["relation"] == "different_scope" and v["dimension"] == "attribution"

    def test_monthly_vs_annual_is_different_scope_not_figure_contradiction(self):
        c = self._f(measure="التضخم الشهري", population=None, time_period="سبتمبر 2026",
                    polarity="up", numbers=["1.5%"])
        s = self._f(time_period="سبتمبر 2026", numbers=["13.9%"])
        v = compare_facets(c, s)
        assert v["relation"] != "contradicts"
