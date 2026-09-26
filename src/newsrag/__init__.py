"""NewsRAG: research with an inspectable evidence trail."""
from .asyncio import AsyncNewsroomRAG
from .core import Answer, Evidence, NewsroomRAG, Source, ValidationError
from .evaluation import RetrievalCase, RetrievalScore, evaluate_retrieval
from .ingest import source_from_html, source_from_pdf
from .storage import load_sources, save_sources

__all__ = ["Answer", "AsyncNewsroomRAG", "Evidence", "NewsroomRAG", "RetrievalCase", "RetrievalScore", "Source", "ValidationError", "evaluate_retrieval", "load_sources", "save_sources", "source_from_html", "source_from_pdf"]
__version__ = "0.2.0"
