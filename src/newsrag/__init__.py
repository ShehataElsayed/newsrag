"""NewsRAG: research with an inspectable evidence trail."""
from .asyncio import AsyncNewsroomRAG
from .core import Answer, Evidence, NewsroomRAG, Source, ValidationError
from .evaluation import RetrievalCase, RetrievalScore, evaluate_retrieval
from .ingest import source_from_html, source_from_pdf
from .sqlite import SQLiteSourceStore
from .storage import load_sources, save_sources

__all__ = ["Answer", "AsyncNewsroomRAG", "Evidence", "NewsroomRAG", "RetrievalCase", "RetrievalScore", "SQLiteSourceStore", "Source", "ValidationError", "evaluate_retrieval", "load_sources", "save_sources", "source_from_html", "source_from_pdf"]
__version__ = "0.3.0"
