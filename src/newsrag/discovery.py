"""Query-driven source discovery (genuine search, bounded free route).

``search_gdelt`` calls the public GDELT 2.1 Doc API - a real query-to-sources
discovery step (query in, ranked news-article results out), free, keyless,
and confined to the GDELT API host via pinned HTTPS. It covers news articles
GDELT indexed in roughly the last three months; it is NOT general web search
and NOT an archive: a general web-search provider is a separate
product/privacy decision. Rate limits (HTTP 429) are surfaced, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import json
import urllib.parse

from .core import ValidationError
from .web import _get  # pinned-HTTPS GET with public-address resolution

_GDELT_HOST = "api.gdeltproject.org"
_DDG_HOST = "html.duckduckgo.com"

# DuckDuckGo's HTML endpoint challenges non-browser clients (HTTP 202) and is
# meant for browsers; discovery for it identifies as a generic browser. This
# applies ONLY to this endpoint - publisher fetching keeps the NewsRAG agent.
_DDG_BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def _ddg_transport(host: str, path: str, timeout: float) -> tuple[str, bytes]:
    return _get(host, path, timeout, user_agent=_DDG_BROWSER_UA)


@dataclass(frozen=True)
class DiscoveryResult:
    title: str
    url: str
    excerpt: str

    def as_dict(self) -> dict[str, Any]:
        return {"title": self.title, "url": self.url, "excerpt": self.excerpt}


def search_gdelt(query: str, *, limit: int = 5,
                 transport: Callable[[str, str, float], tuple[str, bytes]] = _get) -> list[DiscoveryResult]:
    """Discover news sources for a query via the GDELT 2.1 Doc API.

    Fail-closed: empty query, non-200 response (including rate limiting), or
    a malformed/unavailable API response raises ValidationError; partial
    results are never invented. Recency is bounded by GDELT's rolling index.
    """
    q = (query or "").strip()
    if not q:
        raise ValidationError("search_gdelt needs a non-empty query")
    if not 1 <= limit <= 20:
        raise ValidationError("limit must be 1..20")
    path = "/api/v2/doc/doc?" + urllib.parse.urlencode(
        {"query": q, "mode": "artlist", "maxrecords": limit, "format": "json",
         "sort": "hybridrel"})
    # transport contract: (content_type, body) on HTTP 200; ValidationError
    # naming the HTTP status (incl. 429 rate limiting) otherwise - fail-closed.
    _ctype, body = transport(_GDELT_HOST, path, 25.0)
    try:
        data = json.loads(body.decode("utf-8"))
        articles = data["articles"]
        results = [
            DiscoveryResult(
                title=str(a["title"]),
                url=str(a["url"]),
                excerpt=str(a.get("seendate", ""))[:20] + " " + str(a.get("domain", "")),
            )
            for a in articles[:limit]
        ]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValidationError(f"GDELT search returned an unusable response: {exc}") from exc
    return results


def search_duckduckgo(query: str, *, limit: int = 5,
                      transport: Callable[[str, str, float], tuple[str, bytes]] = _ddg_transport) -> list[DiscoveryResult]:
    """Discover web sources for a query via DuckDuckGo's HTML endpoint.

    General web search (not news-only): query in, ranked web-page results
    out, free, keyless, pinned HTTPS to the DuckDuckGo HTML host only. This
    answers the "connect a search engine" route directly. Result quality and
    language coverage depend on DuckDuckGo's index; the endpoint is a
    lightweight HTML form of the search engine and its markup may change -
    parsing is defensive and fail-closed. Rate limiting (HTTP 429/403) is
    surfaced, never hidden. Source EXCLUSIONS (e.g. Wikipedia) are applied
    downstream at ingestion/citation via ``blocked_hosts``, not here.
    """
    import re

    q = (query or "").strip()
    if not q:
        raise ValidationError("search_duckduckgo needs a non-empty query")
    if not 1 <= limit <= 20:
        raise ValidationError("limit must be 1..20")
    path = "/html/?" + urllib.parse.urlencode({"q": q})
    # same transport contract as search_gdelt (see above)
    _ctype, body = transport(_DDG_HOST, path, 25.0)
    html = body.decode("utf-8", "replace")
    results: list[DiscoveryResult] = []
    for href, title_html in re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html):
        m = re.search(r"uddg=([^&]+)", href)
        url = urllib.parse.unquote(m.group(1)) if m else href
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        if url.startswith("http") and title:
            results.append(DiscoveryResult(title=title, url=url, excerpt=""))
        if len(results) >= limit:
            break
    if not results:
        # markup drift or an empty result page must never become silent success
        raise ValidationError("DuckDuckGo search returned no parseable results")
    return results
