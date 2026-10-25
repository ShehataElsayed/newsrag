"""NewsRAG: research with an inspectable evidence trail."""
from .arabic import ArabicLightTokenizer
from .assist import AssistResult, ClaimSupport, assist
from .asyncio import AsyncNewsroomRAG
from .core import Answer, Evidence, NewsroomRAG, Source, ValidationError
from .gemini import GeminiProvider
from .evaluation import RetrievalCase, RetrievalScore, evaluate_retrieval
from .ingest import source_from_html, source_from_pdf
from .multilingual import MultilingualEmbedder
from .sqlite import SQLiteSourceStore
from .storage import load_sources, save_sources
from .web import source_from_url

__all__ = ["Answer", "AssistResult", "ClaimSupport", "assist", "ArabicLightTokenizer", "AsyncNewsroomRAG", "Evidence", "GeminiProvider", "MultilingualEmbedder", "NewsroomRAG", "RetrievalCase", "RetrievalScore", "SQLiteSourceStore", "Source", "ValidationError", "evaluate_retrieval", "load_sources", "save_sources", "source_from_html", "source_from_pdf", "source_from_url"]
__version__ = "0.6.0"
