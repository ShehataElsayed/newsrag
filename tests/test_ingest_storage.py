import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from newsrag import (
    NewsroomRAG,
    Source,
    ValidationError,
    load_sources,
    save_sources,
    source_from_html,
    source_from_pdf,
)


class TestIngestStorage(unittest.TestCase):
    def test_html_removes_scripts_and_retains_text(self):
        src = source_from_html(id="w", title="Page", url="https://example.org/a",
                               html="<h1>خبر</h1><script>ignore me</script><p>النص</p>")
        self.assertIn("خبر", src.text)
        self.assertIn("النص", src.text)
        self.assertNotIn("ignore me", src.text)
        self.assertEqual(src.source_type, "webpage")

    def test_html_url_validation(self):
        with self.assertRaises(ValidationError):
            source_from_html(id="w", title="Page", url="file:///etc/passwd", html="<p>text</p>")

    def test_snapshot_round_trip(self):
        src = Source("arabic", "بيان", "أعلنت الشركة", published_at=datetime(2026, 9, 1, tzinfo=timezone.utc))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sources.json"
            save_sources(NewsroomRAG().add(src), path)
            loaded = load_sources(path)
            self.assertEqual(loaded.search("الشركة")[0].source_id, "arabic")
            self.assertEqual(loaded._sources["arabic"].published_at, src.published_at)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_snapshot_schema_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"schema": 99, "sources": []}')
            with self.assertRaises(ValidationError):
                load_sources(path)

    def test_pdf_bad_header(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError):
                source_from_pdf(id="p", title="PDF", pdf_bytes=b"not pdf")
        else:
            with self.assertRaises(ValidationError):
                source_from_pdf(id="p", title="PDF", pdf_bytes=b"not pdf")
