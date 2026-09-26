# ruff: noqa: SIM117
import socket
import unittest
from unittest.mock import patch

from newsrag import ValidationError
from newsrag.web import _allowed_url, _public_address, source_from_url

HOSTS = frozenset({"example.org"})


class TestWeb(unittest.TestCase):
    def test_url_restrictions(self):
        for url in ("http://example.org/a", "https://localhost/a",
                    "https://example.org:8443/a", "https://user:pass@example.org/a",
                    "https://example.org.evil.test/a"):
            with self.assertRaises(ValidationError):
                _allowed_url(url, HOSTS)
        self.assertEqual(_allowed_url("https://example.org/a?q=1", HOSTS),
                         ("example.org", "/a?q=1"))

    def test_private_dns_rejected(self):
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, None, None, None,
                                                        ("127.0.0.1", 443))]):
            with self.assertRaises(ValidationError):
                _public_address("example.org")

    def test_robots_and_extraction(self):
        def fetch(host, path, timeout):
            if path == "/robots.txt":
                return "text/plain", b"User-agent: *\nDisallow: /private\n"
            return "text/html", b"<html><article><p>Actual article text.</p></article></html>"
        with self.assertRaises(ValidationError):
            source_from_url(id="a", title="A", url="https://example.org/private",
                            allowed_hosts=HOSTS, fetch=fetch)
        with patch("importlib.import_module") as module:
            module.return_value.extract.return_value = "Actual article text."
            s = source_from_url(id="a", title="A", url="https://example.org/story",
                                allowed_hosts=HOSTS, fetch=fetch)
        self.assertEqual(s.text, "Actual article text.")
        self.assertEqual(s.url, "https://example.org/story")

    def test_crawl_delay_requires_external_scheduling(self):
        def fetch(host, path, timeout):
            return "text/plain", b"User-agent: NewsRAG\nCrawl-delay: 60\n" if path == "/robots.txt" else b""
        with self.assertRaises(ValidationError):
            source_from_url(id="a", title="A", url="https://example.org/story",
                            allowed_hosts=HOSTS, fetch=fetch)

    def test_robots_unavailable_is_denied(self):
        with self.assertRaises(ValidationError):
            source_from_url(id="a", title="A", url="https://example.org/story",
                            allowed_hosts=HOSTS, fetch=lambda *args: ("text/html", b""))
