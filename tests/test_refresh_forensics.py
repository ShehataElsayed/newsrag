import os
import tempfile
import unittest

from newsrag import Source, ValidationError
from newsrag.forensics import INDICATORS_NOT_PROOF, inspect_media
from newsrag.refresh import RefreshableStore


def s(i, text):
    return Source(id=i, title=f"عنوان {i}", text=text)

class RefreshTests(unittest.TestCase):
    def test_refresh_adds_and_reports(self):
        with tempfile.TemporaryDirectory() as d, RefreshableStore(os.path.join(d, "s.db")) as st:
                rep = st.refresh([s("a", "نص أول طويل بما يكفي"), s("b", "نص ثان طويل بما يكفي")])
                self.assertEqual((rep.received, rep.added, rep.updated), (2, 2, 0))
                self.assertTrue(rep.as_dict()["prototype"])
    def test_refresh_updates_existing_and_dedupes_content(self):
        with tempfile.TemporaryDirectory() as d, RefreshableStore(os.path.join(d, "s.db")) as st:
                st.refresh([s("a", "نفس النص تماما بما يكفي")])
                rep = st.refresh([s("a", "نص محدث مختلف بما يكفي"), s("b", "نفس النص تماما بما يكفي")])
                self.assertEqual((rep.updated, rep.added, rep.skipped_duplicates), (1, 0, 1))
    def test_refresh_rejects_invalid_items_by_name(self):
        with tempfile.TemporaryDirectory() as d, RefreshableStore(os.path.join(d, "s.db")) as st:
                rep = st.refresh(["not-a-source"])
                self.assertEqual(rep.added, 0)
                self.assertEqual(len(rep.rejected), 1)

class ForensicsTests(unittest.TestCase):
    def test_missing_file_fails_closed(self):
        with self.assertRaises(ValidationError):
            inspect_media("/nonexistent.jpg")
    def test_png_indicators(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # minimal PNG header
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            path = f.name
        try:
            r = inspect_media(path, include_exif=False)
            self.assertEqual(r.media_type, "png")
            self.assertEqual(len(r.sha256), 64)
            self.assertEqual(r.as_dict()["caveat"], INDICATORS_NOT_PROOF)
        finally:
            os.unlink(path)
