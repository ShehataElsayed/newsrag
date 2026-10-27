"""Optional local multilingual sentence embeddings for semantic retrieval."""
from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualEmbedder:
    """Lazy local model adapter. First use may download model weights.

    Pass a preloaded model to avoid network access at construction and to pin a
    reviewed model revision. Do not send private text to unknown model services.
    """

    def __init__(self, model_name: str = MODEL, *, model: Any = None,
                 device: str | None = None, document_prefix: str = "",
                 query_prefix: str = ""):
        if not isinstance(document_prefix, str) or not isinstance(query_prefix, str):
            raise TypeError("Embedding prefixes must be strings")
        self.document_prefix, self.query_prefix = document_prefix, query_prefix
        self.model_name = model_name
        self._model = model
        self.device = device

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, self.document_prefix)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Encode queries with an explicitly chosen model-specific prefix.

        For E5, configure document_prefix="passage: " and query_prefix="query: ".
        Prefixes are not guessed from model names. Document excerpts stay raw.
        """
        return self._encode(texts, self.query_prefix)

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            try:
                SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer
            except ImportError as exc:
                raise ImportError('Install the optional extra: pip install "newsrag[multilingual]"') from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        vectors: Sequence[Sequence[float]] = self._model.encode(
            [prefix + text for text in texts], convert_to_numpy=True, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]
