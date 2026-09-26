"""NewsRAG: research with an inspectable evidence trail."""
from .core import Answer, Evidence, NewsroomRAG, Source, ValidationError
from .ingest import source_from_html, source_from_pdf
from .storage import load_sources, save_sources

__all__ = ["Answer", "Evidence", "NewsroomRAG", "Source", "ValidationError", "load_sources", "save_sources", "source_from_html", "source_from_pdf"]
__version__ = "0.2.0"
