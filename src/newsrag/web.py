"""Conservative HTTPS article fetching with host allowlisting and robots checks.

This is not a crawler: callers select permitted publisher hosts and article URLs.
"""
from __future__ import annotations

import http.client
import importlib
import ipaddress
import math
import socket
import ssl
from collections.abc import Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from .core import Source, ValidationError

_USER_AGENT = "NewsRAG/0.3 (+https://github.com/ShehataElsayed/newsrag)"
_MAX_BYTES = 2_000_000


def _allowed_url(url: str, hosts: frozenset[str]) -> tuple[str, str]:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if (parsed.scheme != "https" or host not in hosts or parsed.port not in (None, 443)
            or parsed.username or parsed.password or not host):
        raise ValidationError("Only explicitly allowlisted HTTPS publisher hosts are accepted")
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return host, path


def _public_address(host: str) -> str:
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValidationError("Publisher DNS must resolve only to public addresses")
    return str(addresses[0][4][0])


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host: str, address: str, timeout: float):
        super().__init__(host, timeout=timeout, context=ssl.create_default_context())
        self._address = address

    def connect(self) -> None:
        raw = socket.create_connection((self._address, 443), self.timeout)
        try:
            self.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=self.host)
        except Exception:
            raw.close()
            raise


def _get(host: str, path: str, timeout: float) -> tuple[str, bytes]:
    address = _public_address(host)
    connection = _PinnedHTTPS(host, address, timeout)
    try:
        connection.request("GET", path, headers={"User-Agent": _USER_AGENT,
                                                   "Accept-Encoding": "identity",
                                                   "Host": host})
        response = connection.getresponse()
        if response.status != 200:
            raise ValidationError(f"Publisher returned HTTP {response.status}; redirects are not followed")
        content_type = response.getheader("Content-Type", "").lower()
        data = response.read(_MAX_BYTES + 1)
        if len(data) > _MAX_BYTES:
            raise ValidationError("Publisher response exceeds 2 MB")
        return content_type, data
    finally:
        connection.close()


def source_from_url(*, id: str, title: str, url: str, allowed_hosts: frozenset[str],
                    timeout: float = 10, fetch: Callable[[str, str, float], tuple[str, bytes]] = _get) -> Source:
    """Fetch a single publisher article only after a permitted robots.txt check.

    Restrict allowed_hosts to publishers reviewed by the application. Redirects
    and private IPs are rejected; page authorship still requires human review.
    """
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 30:
        raise ValidationError("Timeout must be between 0 and 30 seconds")
    host, path = _allowed_url(url, allowed_hosts)
    robots_type, robots_bytes = fetch(host, "/robots.txt", timeout)
    if "text/plain" not in robots_type or not robots_bytes:
        raise ValidationError("Cannot verify publisher robots.txt")
    robots = RobotFileParser()
    robots.parse(robots_bytes.decode("utf-8", errors="replace").splitlines())
    if not robots.can_fetch(_USER_AGENT, url):
        raise ValidationError("Publisher robots.txt disallows this URL")
    delay = robots.crawl_delay(_USER_AGENT)
    if delay and float(delay) > 0:
        raise ValidationError("Publisher specifies a crawl delay; schedule this fetch externally")
    article_type, article = fetch(host, path, timeout)
    if "text/html" not in article_type:
        raise ValidationError("Publisher response is not HTML")
    try:
        extract = importlib.import_module("trafilatura").extract
    except ImportError as exc:
        raise ImportError('Install the optional extra: pip install "newsrag[web]"') from exc
    text = extract(article.decode("utf-8", errors="replace"), include_comments=False,
                   include_tables=False, url=url, favor_precision=True)
    if not text or not text.strip():
        raise ValidationError("No article text extracted; inspect the page manually")
    return Source(id=id, title=title, text=text.strip(), url=url, source_type="webpage")
