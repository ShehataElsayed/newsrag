"""Optional Snowball Arabic light stemming for lexical search.

Light stemming is not full morphological analysis or contextual lemmatization.
"""
from __future__ import annotations

import importlib
import unicodedata

from .core import _ARABIC_MARKS, _WORD, _terms

_ARABIC_TRANSLATION = str.maketrans("أإآٱىؤئةکی", "اااايويهكي")


class ArabicLightTokenizer:
    """Normalize Arabic spelling, then add Snowball stems alongside surface tokens."""

    def __init__(self) -> None:
        try:
            self.stemmer = importlib.import_module("snowballstemmer").stemmer("arabic")
        except ImportError as exc:
            raise ImportError('Install the optional extra: pip install "newsrag[arabic]"') from exc

    def __call__(self, text: str) -> list[str]:
        tokens = _terms(text)
        result = list(tokens)
        originals = _WORD.findall(_ARABIC_MARKS.sub("", unicodedata.normalize("NFKC", text.lower())))
        for token, original in zip(tokens, originals):
            if any("\u0621" <= char <= "\u064a" for char in token):
                # Snowball expects original Arabic spelling before NewsRAG's
                # ta-marbuta normalization; then normalize the stem for matching.
                stem = self.stemmer.stemWord(original)
                stem = stem.translate(_ARABIC_TRANSLATION)
                if stem != token:
                    result.append(stem)
        return result
