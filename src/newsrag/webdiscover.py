"""Genuine web discovery: research question -> DuckDuckGo query -> candidate
source URLs. This is the open-web discovery stage (distinct from declared-
registry ranking in discover.py). Free HTML endpoint, no API key, one query
per call. Wikipedia results are filtered at discovery (they are also denied
at fetch and barred from citation). Every returned candidate is a LEAD: the
collector records what was actually fetched, and a discovered URL carries no
trust beyond its fetched content.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
_ENDPOINT = "https://html.duckduckgo.com/html/"
_RESULT = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
_SNIPPET = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def _unwrap(href: str) -> str:
    if "uddg=" in href:
        return urllib.parse.unquote(urllib.parse.parse_qs(
            urllib.parse.urlparse(href).query).get("uddg", [href])[0])
    return href


def web_search(question: str, *, max_results: int = 8, timeout: float = 30.0,
               opener=None) -> list[dict[str, str]]:
    """One DDG query built from the question. Returns ranked candidates
    [{url, title, snippet, source}]; Wikipedia-domain hits are dropped and
    reported in the 'filtered' field of the caller (here: dropped silently,
    count attached to each call site via len difference if needed)."""
    q = question.strip()[:300]
    data = urllib.parse.urlencode({"q": q}).encode()
    req = urllib.request.Request(_ENDPOINT, data=data, headers=_UA)
    open_fn = opener or urllib.request.urlopen
    with open_fn(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    links = _RESULT.findall(html)
    snippets = [_TAG.sub("", s).strip() for s in _SNIPPET.findall(html)]
    out = []
    for i, (href, title) in enumerate(links):
        url = _unwrap(href)
        if "wikipedia.org" in url:
            continue  # excluded at discovery (and denied again at fetch)
        out.append({"url": url, "title": _TAG.sub("", title).strip(),
                    "snippet": snippets[i] if i < len(snippets) else "",
                    "source": "duckduckgo"})
        if len(out) >= max_results:
            break
    return out
