import json
import unittest

import pytest
from newsrag.discovery import search_duckduckgo, search_gdelt, DiscoveryResult
from newsrag.core import ValidationError


def fake_transport(articles, status="200"):
    def _t(host, path, timeout):
        assert host == "api.gdeltproject.org"
        if status != "200":
            raise ValidationError(f"Publisher returned HTTP {status}; redirects are not followed")
        return "application/json", json.dumps({"articles": articles}).encode()
    return _t


def test_search_gdelt_returns_results():
    arts = [{"title": "خبر", "url": "https://news.example/a1", "seendate": "20261025T120000Z", "domain": "news.example"}]
    out = search_gdelt("القاهرة", transport=fake_transport(arts))
    assert out == [DiscoveryResult(title="خبر", url="https://news.example/a1",
                                   excerpt="20261025T120000Z news.example")]


def test_search_gdelt_rejects_empty_query():
    with pytest.raises(ValidationError):
        search_gdelt("   ", transport=fake_transport([]))


def test_search_gdelt_rejects_bad_limit():
    with pytest.raises(ValidationError):
        search_gdelt("x", limit=0, transport=fake_transport([]))


def test_search_gdelt_surfaces_rate_limit():
    with pytest.raises(ValidationError, match="429"):
        search_gdelt("x", transport=fake_transport([], status="429"))

def test_search_gdelt_fails_closed_on_http_error():
    with pytest.raises(ValidationError, match="503"):
        search_gdelt("x", transport=fake_transport([], status="503"))


def test_search_gdelt_fails_closed_on_malformed_response():
    def bad(host, path, timeout):
        return "200", b"not json"
    with pytest.raises(ValidationError):
        search_gdelt("x", transport=bad)


def test_search_gdelt_empty_articles_is_empty_result():
    assert search_gdelt("x", transport=fake_transport([])) == []


class DuckDuckGoSearchTests(unittest.TestCase):
    """search_duckduckgo: genuine general-web discovery, fail-closed."""

    HTML = (
        '<a rel="nofollow" class="result__a" '
        'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews%2Fx&amp;rut=ab">'
        'مثال <b>لنتيجة</b></a>'
        '<a rel="nofollow" class="result__a" '
        'href="https://second.example/article">Second title</a>'
    )

    def test_parses_results_and_unwraps_redirects(self):
        def fake(host, path, timeout):
            assert host == "html.duckduckgo.com"
            assert "q=" in path
            return "text/html", self.HTML.encode()
        rs = search_duckduckgo("عاصمة مصر", transport=fake)
        self.assertEqual([r.url for r in rs],
                         ["https://example.com/news/x", "https://second.example/article"])
        self.assertEqual(rs[0].title, "مثال لنتيجة")

    def test_rate_limit_is_an_error_not_silence(self):
        def blocked(host, path, timeout):
            raise ValidationError("Publisher returned HTTP 429; redirects are not followed")
        with self.assertRaises(ValidationError):
            search_duckduckgo("x", transport=blocked)

    def test_unparseable_markup_fails_closed(self):
        with self.assertRaises(ValidationError):
            search_duckduckgo("x", transport=lambda h, p, t: ("text/html", b"<html>no results here</html>"))

    def test_empty_query_rejected(self):
        with self.assertRaises(ValidationError):
            search_duckduckgo("  ", transport=lambda h, p, t: ("text/html", b""))
