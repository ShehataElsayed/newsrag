"""Hybrid source-store refresh interface (bounded local prototype).

This is the ingestion/refresh half of the "hybrid auto-updating DB" idea,
deliberately scoped to a PROTOTYPE: the caller supplies sources (files,
loader results, or an explicitly user-chosen feed) and this module ingests
them into a local SQLite store with dedupe, provenance, and refresh metadata.
It does NOT run a sustained service, does NOT pick feeds by itself, and sets
up NO permanent external monitor. A production auto-updating deployment
(source/licensing choices, scheduling, hosting) is a separate decision.

Refresh semantics: re-ingesting a source id replaces its row (upsert) and
updates its ``accessed_at``; ``RefreshReport`` records exactly what changed so
refreshes are auditable rather than silent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .core import Source, ValidationError
from .sqlite import SQLiteSourceStore

PROTOTYPE = True  # bounded local prototype, not a sustained service


@dataclass(frozen=True)
class RefreshReport:
    """Auditable outcome of one refresh run."""

    received: int
    added: int
    updated: int
    skipped_duplicates: int
    rejected: tuple[str, ...]
    refreshed_at: str

    def as_dict(self) -> dict:
        return {"received": self.received, "added": self.added, "updated": self.updated,
                "skipped_duplicates": self.skipped_duplicates, "rejected": list(self.rejected),
                "refreshed_at": self.refreshed_at, "prototype": PROTOTYPE}


def content_hash(source: Source) -> str:
    """Stable TEXT fingerprint for duplicate detection. Text only, not title:
    syndicated pieces republished under different headlines are duplicates."""
    h = hashlib.sha256()
    h.update(source.text.strip().encode("utf-8"))
    return h.hexdigest()


class RefreshableStore:
    """Local store with an explicit, auditable refresh path."""

    def __init__(self, path: str | Path):
        self._store = SQLiteSourceStore(path)

    def refresh(self, sources: Iterable[Source], *, skip_content_duplicates: bool = True) -> RefreshReport:
        """Ingest caller-supplied sources. Fail-closed on invalid items: they are
        rejected and named in the report, never silently dropped or ingested."""
        existing = {s.id: s for s in self._store.sources()}
        existing_hashes = {content_hash(s) for s in existing.values()}
        received = added = updated = dupes = 0
        rejected: list[str] = []
        now = datetime.now(timezone.utc)
        for src in sources:
            received += 1
            try:
                if not isinstance(src, Source):
                    raise ValidationError("refresh items must be Source objects")
                if skip_content_duplicates and src.id not in existing and content_hash(src) in existing_hashes:
                    dupes += 1
                    continue
                if src.accessed_at is None:
                    src = Source(id=src.id, title=src.title, text=src.text, url=src.url,
                                 published_at=src.published_at, accessed_at=now,
                                 publisher=src.publisher, language=src.language,
                                 source_type=src.source_type)
                self._store.upsert(src)
                if src.id in existing:
                    updated += 1
                else:
                    added += 1
            except (ValidationError, ValueError) as exc:
                rejected.append(f"{getattr(src, 'id', '?')}: {exc}")
        return RefreshReport(received, added, updated, dupes, tuple(rejected), now.isoformat())

    def sources(self):
        return self._store.sources()

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> RefreshableStore:  # noqa: PYI034 - returns self at runtime
        return self

    def __exit__(self, *exc) -> None:
        self.close()
