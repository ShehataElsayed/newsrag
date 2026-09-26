"""Transactional SQLite source store, independent of provider-specific vectors."""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from .core import Source, ValidationError

_SCHEMA = 1


class SQLiteSourceStore:
    """Persist source text and metadata, not embeddings or provider credentials.

    Rebuild a RAG index from `sources()` when starting a process. Back up the
    SQLite file using SQLite's backup API, not by copying a live database.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(self.path) == ":memory:":
            raise ValidationError("SQLiteSourceStore requires a file path")
        new_file = not self.path.exists()
        # The default SQLite create mode may expose private newsroom sources.
        if new_file:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(fd)
        self._db = sqlite3.connect(self.path)
        try:
            with self._db:
                self._db.execute("PRAGMA user_version")
                version = self._db.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, _SCHEMA):
                    raise ValidationError("Unsupported SQLite source schema")
                if version == 0:
                    self._db.execute("CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                    self._db.execute(f"PRAGMA user_version = {_SCHEMA}")
        except Exception:
            self._db.close()
            raise

    def backup(self, destination: str | Path) -> None:
        """Take a consistent SQLite backup, including uncheckpointed live changes."""
        target = Path(destination)
        if target == self.path:
            raise ValidationError("Backup destination must differ from the source")
        if target.exists():
            raise ValidationError("Backup destination already exists")
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(fd)
        try:
            with sqlite3.connect(target) as output:
                self._db.backup(output)
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def upsert(self, source: Source) -> None:
        """Insert or replace one source atomically."""
        record = {"id": source.id, "title": source.title, "text": source.text,
                  "url": source.url, "published_at": source.published_at.isoformat() if source.published_at else None,
                  "accessed_at": source.accessed_at.isoformat() if source.accessed_at else None,
                  "publisher": source.publisher, "language": source.language,
                  "source_type": source.source_type}
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO sources (id,payload) VALUES (?,?)",
                             (source.id, json.dumps(record, ensure_ascii=False)))

    def delete(self, source_id: str) -> bool:
        with self._db:
            result = self._db.execute("DELETE FROM sources WHERE id=?", (source_id,))
            return result.rowcount > 0

    def sources(self) -> Iterator[Source]:
        for (payload,) in self._db.execute("SELECT payload FROM sources ORDER BY id"):
            record = json.loads(payload)
            for key in ("published_at", "accessed_at"):
                if record[key] is not None:
                    record[key] = datetime.fromisoformat(record[key])
            yield Source(**record)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> SQLiteSourceStore:  # noqa: PYI034
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
