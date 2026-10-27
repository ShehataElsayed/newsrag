"""Question-directed source collector: declared sources only, never a crawler.

An investigation question names WHAT to find; a SourceSpec registry names WHERE
the library is allowed to look. The collector fetches only URLs that are either
declared directly by the investigator or produced by a registered adapter for a
declared source. Every attempt - success or failure - becomes a Provenance
record (url, fetch time, publisher vs speaker, reference period, status), so a
report can show exactly what was gathered, from where, and what failed.

Workflow rules:
- Wikipedia is denied at discovery, ingestion, and citation time.
- Absence of evidence in a fetch is recorded, never treated as negation.
- Collection is distinct from evidence support: fetching a page does not make
  it support any claim (see newsrag.semantic for the support check).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from .core import ValidationError
from .web import _allowed_url, _get

DENYLIST = ("wikipedia.org", "wikiwand.com")



@dataclass
class SourceSpec:
    """A declared source the question directs the collector to."""
    name: str                      # human label, e.g. "AFH deport-Egyptians investigation"
    urls: tuple[str, ...]          # declared URLs (no crawling beyond these)
    publisher: str                 # who publishes the page (NOT necessarily who said the fact)
    reference_period: str | None = None   # period the data is ABOUT, if known
    kind: str = "html"             # html | json | pdf
    topics: str = ""               # registry-owner keywords for discovery matching


@dataclass
class Provenance:
    """Auditable record of one fetch attempt."""
    source: str
    url: str
    fetched_at: str                # UTC ISO timestamp of THIS attempt
    publisher: str
    reference_period: str | None
    status: str                    # ok | failed | denied
    detail: str = ""               # error text or byte count
    speaker: str | None = None     # who the page attributes facts to, when distinct from publisher
    raw_sha256: str | None = None
    text_sha256: str | None = None
    extraction: str | None = None
    transport: str = "pinned_https"


def _denied(url: str) -> bool:
    try:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return True
    return any(host == d or host.endswith("." + d) for d in DENYLIST)


def fetch_url(url: str, *, timeout: float = 30.0, opener: Callable | None = None) -> bytes:
    """One explicit HTTPS URL, robots permitted, public DNS, 2MB maximum.

    Production uses the existing DNS-pinned transport, which rejects redirects.
    An injected opener is a caller-controlled testing/transport seam, not proof
    of network protections or publisher identity. It must enforce public DNS
    and no redirects itself; it is labelled as injected in provenance.
    """
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 0 < timeout <= 30):
        raise ValidationError("Timeout must be between 0 and 30 seconds")
    if _denied(url):
        raise ValidationError("Source host excluded by workflow policy")
    try:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
        host, path = _allowed_url(url, frozenset({host}))
    except ValueError as exc:
        raise ValidationError("Invalid collector URL") from exc

    def get(target: str) -> tuple[str, bytes]:
        if opener is None:
            return _get(host, target, timeout)
        req = urllib.request.Request(f"https://{host}{target}", headers={
            "User-Agent": "NewsRAG/0.3 (+https://github.com/ShehataElsayed/newsrag)"})
        with opener(req, timeout=timeout) as resp:
            # Test transports may omit headers/status, never treated as live proof.
            status = getattr(resp, "status", 200)
            final_url = resp.geturl() if hasattr(resp, "geturl") else req.full_url
            if status != 200 or final_url != req.full_url:
                raise ValidationError("Redirect or non-200 collector response rejected")
            raw = resp.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValidationError("Collector response exceeds 2 MB")
            headers = getattr(resp, "headers", {})
            return headers.get("Content-Type", "text/plain" if target == "/robots.txt" else "text/html"), raw

    try:
        kind, robots_raw = get("/robots.txt")
    except Exception as exc:
        raise ValidationError(f"robots check failed: {type(exc).__name__}") from exc
    if "text/plain" not in kind or not robots_raw:
        raise ValidationError("Cannot verify publisher robots.txt")
    robot = RobotFileParser()
    robot.parse(robots_raw.decode("utf-8", errors="replace").splitlines())
    agent = "NewsRAG/0.3 (+https://github.com/ShehataElsayed/newsrag)"
    if not robot.can_fetch(agent, url):
        raise ValidationError("Publisher robots.txt disallows collector URL")
    if robot.crawl_delay(agent):
        raise ValidationError("Publisher crawl delay requires external scheduling")
    try:
        kind, raw = get(path)
    except Exception as exc:
        raise ValidationError(f"article fetch failed: {type(exc).__name__}") from exc
    if not any(t in kind.lower() for t in ("text/html", "application/json", "application/pdf",
                                          "text/plain", "application/xhtml+xml")):
        raise ValidationError("Unsupported collector content type")
    return raw


_META_RE = re.compile(
    r'<meta[^>]+(?:name|property)=["\'](?:description|og:description|og:title)["\']'
    r'[^>]+content=["\'](.*?)["\']'
    r'|<meta[^>]+content=["\'](.*?)["\']'
    r'[^>]+(?:name|property)=["\'](?:description|og:description|og:title)["\']',
    re.IGNORECASE | re.DOTALL)


def html_to_text2(raw: bytes) -> tuple[str, bool]:
    """Returns (text, used_meta_fallback). JS-shell pages (SPA/GASTAT-style):
    the visible body renders client-side, but the server still ships the
    publisher-authored summary in meta description/og tags. When the body is
    thin, those tags supplement the fetched document - same URL, same
    provenance, but the caller must label it a summary, not full-page
    coverage ."""
    import html as _html
    t = _html.unescape(raw.decode("utf-8", errors="replace"))
    metas = []
    for m in _META_RE.finditer(t):
        metas.append(m.group(1) or m.group(2) or "")
    t = re.sub(r"(?is)<(script|style|noscript|header|footer|nav)[^>]*>.*?</\1>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    body = re.sub(r"\s+", " ", t).strip()
    used_meta = False
    if len(body) < 500:
        extra = " ".join(re.sub(r"\s+", " ", x).strip() for x in metas if x.strip())
        if extra:
            body = f"{body} {extra}".strip()
            used_meta = True
    return body, used_meta


def html_to_text(raw: bytes) -> str:
    return html_to_text2(raw)[0]


def json_to_text(raw: bytes) -> str:
    """JSON payload -> text. Objects with scalar leaves render as record lines
    ("key: value; key: value") so a label stays bound to its value - flattening
    structure into bare values would orphan figures from what they measure."""
    def _record(o):
        """All scalar leaves of one record, dotted keys, one line: the indicator
        label, geography and date stay attached to the value they describe."""
        scalars: dict[str, object] = {}
        def walk(d, pre):
            for k, v in d.items():
                if isinstance(v, (str, int, float)) and v is not None and str(v) != "":
                    v2 = html_to_text(str(v).encode()) if isinstance(v, str) and "<" in str(v) else v
                    scalars[f"{pre}{k}"] = v2
                elif isinstance(v, dict):
                    walk(v, f"{pre}{k}.")
        walk(o, "")
        return scalars

    def _vals(o):
        if isinstance(o, str):
            yield html_to_text(o.encode()) if "<" in o else o
        elif isinstance(o, dict):
            rec = _record(o)
            if rec:
                yield "; ".join(f"{k}: {v}" for k, v in rec.items())
            for v in o.values():
                if isinstance(v, list):
                    yield from _vals(v)
        elif isinstance(o, list):
            for v in o:
                yield from _vals(v)
    try:
        obj = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return raw.decode("utf-8", errors="replace")
    return "\n".join(v.strip() for v in _vals(obj) if v and v.strip())


def collect(question: str, registry: list[SourceSpec], *,
            opener: Callable | None = None) -> dict[str, Any]:
    """Fetch every declared URL in the registry, for the given question.

    Returns {question, gathered: [{source, url, text}], provenance: [...]}.
    The question is recorded for audit; it directs WHICH registry the caller
    builds, never silent expansion to undeclared URLs.
    """
    gathered: list[dict[str, str]] = []
    provenance: list[Provenance] = []
    for spec in registry:
        for url in spec.urls:
            rec = Provenance(source=spec.name, url=url,
                             fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             publisher=spec.publisher,
                             reference_period=spec.reference_period,
                             status="ok", transport="injected" if opener else "pinned_https")
            if spec.kind not in ("html", "json", "pdf", "text"):
                rec.status, rec.detail = "failed", "Unsupported declared source kind"
                provenance.append(rec)
                continue
            if _denied(url):
                rec.status, rec.detail = "denied", "denylist (Wikipedia excluded at ingestion)"
                provenance.append(rec)
                continue
            try:
                raw = fetch_url(url, opener=opener)
            except Exception as exc:  # noqa: BLE001 - failure is evidence too: record it
                rec.status, rec.detail = "failed", f"{type(exc).__name__}: {exc}"[:300]
                provenance.append(rec)
                continue
            rec.raw_sha256 = hashlib.sha256(raw).hexdigest()
            rec.extraction = spec.kind
            if spec.kind == "html" and raw[:5] != b"%PDF-":
                text, used_meta = html_to_text2(raw)
                if used_meta:
                    rec.extraction = "html_with_meta_summary"
                    rec.detail += "; meta-summary fallback (JS shell: summary, not full-page coverage)"
            elif spec.kind == "json":
                text = json_to_text(raw)
            elif spec.kind == "pdf" or raw[:5] == b"%PDF-":
                # declared statistical publications arrive as PDFs; decode
                # them locally (pypdf), never as garbage "text"
                try:
                    import io

                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(raw))
                    text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
                    # pypdf soft-wraps lines mid-sentence; an orphan fragment
                    # ("approximately 5.56 million by the end of 2025") loses
                    # its head's entity/scope tokens and can never be ranked or
                    # bound correctly. Rejoin lines that do not end with
                    # sentence-ending punctuation.
                    # rejoin soft-wraps WITHIN a paragraph only: a blank-line
                    # break is a real boundary - merging across it glues headers
                    # onto headline sentences and the 400-char candidate cap
                    # drops them (reviewed case: "Total population exceeds 35 million..."
                    # fused to the GASTAT address block).
                    _paras = re.split(r"\n\s*\n", text)
                    text = "\n\n".join(
                        re.sub(r"(?<![.!\u061f?:;»])[ \t]*\n[ \t]*", " ", pg)
                        for pg in _paras)
                    rec.detail += f"; pdf {len(reader.pages)} pages"
                    rec.extraction = "pdf_text"
                except Exception as exc:  # noqa: BLE001 - extract failure recorded as evidence
                    rec.status, rec.detail = "failed", f"pdf extract: {type(exc).__name__}: {exc}"[:200]
                    provenance.append(rec)
                    continue
            else:
                text = raw.decode("utf-8", errors="replace")
            rec.detail = f"{len(raw)} bytes" + rec.detail
            # WAF stub detection: a tiny "Request Rejected" shell is a denial,
            # not content - recording it as ok would let a blocked fetch stand
            # in as evidence silence.
            if len(raw) < 2000 and (b"Request Rejected" in raw or b"Access Denied" in raw
                                    or b"request was denied" in raw.lower()):
                rec.status, rec.detail = "denied", f"WAF rejection stub ({len(raw)} bytes)"
                provenance.append(rec)
                continue
            if not text.strip():
                rec.status, rec.detail = "failed", "No extractable source text"
                provenance.append(rec)
                continue
            rec.text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            provenance.append(rec)
            gathered.append({"source": spec.name, "url": url, "text": text,
                       "reference_period": spec.reference_period or "",
                       "topics": spec.topics or ""})
    return {"question": question,
            "gathered": gathered,
            "provenance": [asdict(p) for p in provenance]}


def save_result(result: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
