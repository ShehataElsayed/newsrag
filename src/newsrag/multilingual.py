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
                 device: str | None = None):
        self.model_name = model_name
        self._model = model
        self.device = device

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            try:
                SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer
            except ImportError as exc:
                raise ImportError('Install the optional extra: pip install "newsrag[multilingual]"') from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        vectors: Sequence[Sequence[float]] = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]
