"""Optional Google Gemini provider adapter (dependency-free, HTTPS via urllib).

Default model names were current stable IDs as of 2026-10-25
(``gemini-3.1-flash-lite`` and ``gemini-embedding-001``, both confirmed against the live model list 2026-10-25; ``gemini-embedding-2`` does not exist and ``gemini-2.5-flash`` is closed to new keys); pass explicit names to pin
or update them.

This module is an adapter, not an endorsement. Text you pass to ``generate`` or
``embed`` leaves your machine and is processed by Google's API under its terms;
only send sources you are allowed to share with a third-party service. The API
key is never logged or included in error messages.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib import request

from .core import ValidationError

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider:
    """Callable ``embed``/``generate`` adapters for :class:`NewsroomRAG`.

    ``transport`` is injectable for tests: it receives a
    ``urllib.request.Request`` and returns the decoded JSON body.
    """

    def __init__(self, api_key: str, *, model: str = "gemini-3.1-flash-lite",
                 embedding_model: str = "gemini-embedding-001",
                 timeout: float = 30.0,
                 transport: Callable[[request.Request], Any] | None = None):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValidationError("GeminiProvider needs a non-empty API key")
        for name, value in (("model", model), ("embedding_model", embedding_model)):
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"{name} must be a non-empty model name")
        self._key = api_key
        self.model, self.embedding_model = model, embedding_model
        self.timeout = timeout
        self._transport = transport or self._post

    def _post(self, req: request.Request) -> Any:
        with request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _call(self, model: str, action: str, payload: dict[str, Any]) -> Any:
        req = request.Request(
            f"{_BASE}/{model}:{action}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json", "x-goog-api-key": self._key},
            method="POST")
        return self._transport(req)

    def generate(self, prompt: str) -> str:
        body = self._call(self.model, "generateContent",
                          {"contents": [{"parts": [{"text": prompt}]}]})
        try:
            parts = body["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValidationError("Gemini generateContent returned an unexpected shape") from exc
        if not text.strip():
            raise ValidationError("Gemini generateContent returned no text")
        return text

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        body = self._call(self.embedding_model, "batchEmbedContents", {
            "requests": [{"model": f"models/{self.embedding_model}",
                          "content": {"parts": [{"text": t}]}} for t in texts]})
        try:
            vectors = [[float(x) for x in e["values"]] for e in body["embeddings"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("Gemini batchEmbedContents returned an unexpected shape") from exc
        if len(vectors) != len(texts) or not vectors[0]:
            raise ValidationError("Gemini embeddings must match the input count")
        return vectors
