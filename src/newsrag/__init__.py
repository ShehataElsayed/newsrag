"""NewsRAG: research with an inspectable evidence trail."""
from .arabic import ArabicLightTokenizer
from .assist import AssistResult, ClaimSupport, TraceStep, assist
from .asyncio import AsyncNewsroomRAG
from .core import Answer, Evidence, NewsroomRAG, Source, ValidationError
from .evaluation import RetrievalCase, RetrievalScore, evaluate_retrieval
from .gemini import GeminiProvider
from .ingest import source_from_html, source_from_pdf
from .multilingual import MultilingualEmbedder
from .qwen import QwenProvider
from .sqlite import SQLiteSourceStore
from .storage import load_sources, save_sources
from .web import source_from_url

__all__ = ["Answer", "ArabicLightTokenizer", "AssistResult", "AsyncNewsroomRAG", "ClaimSupport", "Evidence", "GeminiProvider", "MultilingualEmbedder", "NewsroomRAG", "QwenProvider", "RetrievalCase", "RetrievalScore", "SQLiteSourceStore", "Source", "TraceStep", "ValidationError", "assist", "evaluate_retrieval", "load_sources", "save_sources", "source_from_html", "source_from_pdf", "source_from_url"]
__version__ = "0.6.1"
