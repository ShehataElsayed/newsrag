import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from newsrag import NewsroomRAG, Source, ValidationError
from newsrag.sqlite import SQLiteSourceStore


class TestSQLiteSourceStore(unittest.TestCase):
    def test_round_trip_replace_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "research.db"
            source = Source("a", "عنوان", "إعلان تجريبي", published_at=datetime(2026, 9, 1, tzinfo=timezone.utc))
            with SQLiteSourceStore(path) as db:
                db.upsert(source)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with SQLiteSourceStore(path) as db:
                restored = list(db.sources())
                self.assertEqual(restored, [source])
                self.assertTrue(NewsroomRAG().add(*restored).search("اعلان"))
                db.upsert(Source("a", "New", "Different story"))
                self.assertEqual([s.text for s in db.sources()], ["Different story"])
                self.assertTrue(db.delete("a"))
                self.assertFalse(db.delete("a"))
                self.assertEqual(list(db.sources()), [])

    def test_consistent_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.db"
            copy = Path(directory) / "backup.db"
            with SQLiteSourceStore(path) as db:
                db.upsert(Source("one", "One", "test source"))
                db.backup(copy)
                self.assertEqual(copy.stat().st_mode & 0o777, 0o600)
                with self.assertRaises(ValidationError):
                    db.backup(copy)
            with SQLiteSourceStore(copy) as restored:
                self.assertEqual([s.id for s in restored.sources()], ["one"])

    def test_schema_version_and_in_memory_rejected(self):
        with self.assertRaises(ValidationError):
            SQLiteSourceStore(":memory:")
        with tempfile.TemporaryDirectory() as directory:
            import sqlite3
            path = Path(directory) / "future.db"
            with sqlite3.connect(path) as db:
                db.execute("PRAGMA user_version = 99")
            with self.assertRaises(ValidationError):
                SQLiteSourceStore(path)
