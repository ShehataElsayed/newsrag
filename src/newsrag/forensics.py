"""Opt-in media forensics route (bounded local prototype).

Per the architecture decision this is a SEPARATE, ON-REQUEST route: it is
never called by the RAG/assistant path and its outputs are never fused with
answer verdicts. It reports observable INDICATORS (hashes, container metadata,
EXIF presence) - indicators are not proof: a missing EXIF block, a re-encoded
file, or a matching hash each have innocent explanations, and no output here
establishes authenticity, fabrication, or deepfake status.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .core import ValidationError

INDICATORS_NOT_PROOF = ("Indicators only: metadata presence/absence and hashes never prove "
                        "authenticity or fabrication; results are not fused with any answer verdict.")

_MAX_BYTES = 512 * 1024 * 1024  # refuse absurd inputs, stay bounded


@dataclass(frozen=True)
class MediaIndicators:
    """Observable indicators for one local media file."""

    path: str
    byte_size: int
    sha256: str
    media_type: str
    metadata: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"path": self.path, "byte_size": self.byte_size, "sha256": self.sha256,
                "media_type": self.media_type, "metadata": dict(self.metadata),
                "warnings": list(self.warnings), "caveat": INDICATORS_NOT_PROOF}


def _detect_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:12].find(b"ftyp") >= 0:
        return "mp4-family"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def inspect_media(path: str | Path, *, include_exif: bool = True) -> MediaIndicators:
    """Collect hashes + container/EXIF indicators for one local file.

    Opt-in only: callers must have routed here explicitly. Reading is local;
    nothing is uploaded. EXIF uses Pillow when installed; without it the
    indicator is reported as unavailable rather than guessed.
    """
    p = Path(path)
    if not p.is_file():
        raise ValidationError(f"no such file: {p}")
    size = p.stat().st_size
    if size > _MAX_BYTES:
        raise ValidationError(f"file too large for the bounded inspector: {size} bytes")
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    mtype = _detect_type(data)
    meta: list[tuple[str, str]] = []
    warnings: list[str] = []
    if include_exif and mtype in ("jpeg", "png", "webp"):
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            with Image.open(p) as img:
                exif = img.getexif()
                if exif:
                    for tag_id, value in list(exif.items())[:25]:
                        meta.append((TAGS.get(tag_id, str(tag_id)), str(value)[:120]))
                else:
                    warnings.append("no EXIF block present (common after re-export; not evidence of manipulation)")
        except ImportError:
            warnings.append("Pillow not installed; EXIF indicator unavailable (not inspected)")
    elif include_exif:
        warnings.append(f"EXIF inspection not applicable to detected type: {mtype}")
    return MediaIndicators(str(p), size, sha, mtype, tuple(meta), tuple(warnings))
