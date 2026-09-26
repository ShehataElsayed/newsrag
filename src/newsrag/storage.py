"""Portable JSON snapshot. It is data, not executable code."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from .core import NewsroomRAG, Source, ValidationError

_SCHEMA = 1


def save_sources(rag: NewsroomRAG, path: str | Path) -> None:
    """Atomically save source content and metadata, not provider or embeddings.

    The destination directory must exist. Private material is written with mode 0600.
    """
    destination = Path(path)
    records = [{"id": s.id, "title": s.title, "text": s.text, "url": s.url,
                "published_at": s.published_at.isoformat() if s.published_at else None,
                "accessed_at": s.accessed_at.isoformat() if s.accessed_at else None,
                "publisher": s.publisher, "language": s.language,
                "source_type": s.source_type} for s in rag._sources.values()]
    payload = json.dumps({"schema": _SCHEMA, "sources": records}, ensure_ascii=False)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=".newsrag-", suffix=".tmp", delete=False) as temp:
            name = temp.name
            os.chmod(name, 0o600)
            temp.write(payload)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(name, destination)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def load_sources(path: str | Path, *, rag: NewsroomRAG | None = None) -> NewsroomRAG:
    """Load a trusted local snapshot; embeddings are recomputed by the chosen adapter."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA or not isinstance(payload.get("sources"), list):
        raise ValidationError("Unsupported or invalid NewsRAG snapshot")
    result = rag if rag is not None else NewsroomRAG()
    sources = []
    for record in payload["sources"]:
        if not isinstance(record, dict):
            raise ValidationError("Invalid source record")
        try:
            for key in ("published_at", "accessed_at"):
                if record.get(key) is not None:
                    record[key] = datetime.fromisoformat(record[key])
            sources.append(Source(**record))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid source record") from exc
    return result.add(*sources)
