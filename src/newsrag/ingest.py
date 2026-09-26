"""Explicit, inspectable ingestion adapters. No implicit network requests."""
from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
from io import BytesIO
from urllib.parse import urlparse

from .core import Source, ValidationError


class _TextHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self.hidden += 1
        elif tag in ("p", "div", "article", "h1", "h2", "h3", "li", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self.hidden:
            self.hidden -= 1
        elif tag in ("p", "div", "article", "h1", "h2", "h3", "li"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def source_from_html(*, id: str, title: str, url: str, html: str,
                     published_at: datetime | None = None,
                     publisher: str | None = None) -> Source:
    """Turn caller-fetched HTML into text; caller owns fetching, auth and URL safety."""
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValidationError("A full HTTP(S) source URL is required")
    parser = _TextHTML()
    parser.feed(html)
    text = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    return Source(id=id, title=title, text=text, url=url, published_at=published_at,
                  publisher=publisher, source_type="webpage")


def source_from_pdf(*, id: str, title: str, pdf_bytes: bytes,
                    url: str | None = None, published_at: datetime | None = None,
                    publisher: str | None = None) -> Source:
    """Extract caller-supplied PDF bytes with optional `pypdf` installation.

    Scanned/image-only PDFs need OCR elsewhere; the original PDF remains source of truth.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError('PDF extraction needs `python -m pip install "newsrag[pdf]"`') from exc
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValidationError("Input is not a PDF")
    reader = PdfReader(BytesIO(pdf_bytes))
    text = "\n\n".join(f"[page {index}]\n{page.extract_text() or ''}"
                        for index, page in enumerate(reader.pages, 1))
    if not text.strip() or all(not (page.extract_text() or "").strip() for page in reader.pages):
        raise ValidationError("No extractable text in PDF; OCR may be needed")
    return Source(id=id, title=title, text=text, url=url, published_at=published_at,
                  publisher=publisher, source_type="pdf")
