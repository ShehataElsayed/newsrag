"""Async façade over the synchronous NewsroomRAG core.

The CPU-bound local search is still synchronous; blocking provider calls are run in
worker threads. Do not mutate a shared index concurrently with active operations.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from .core import Answer, Evidence, NewsroomRAG, Source


class AsyncNewsroomRAG:
    """Awaitable adapters for sync or async providers through explicit callables.

    This wrapper does not promise thread safety or transactionality across tasks.
    Use one writer, or coordinate mutations with a lock in the calling application.
    """

    def __init__(self, rag: NewsroomRAG):
        self.rag = rag
        self._lock = asyncio.Lock()

    async def add(self, *sources: Source) -> AsyncNewsroomRAG:
        async with self._lock:
            await asyncio.to_thread(self.rag.add, *sources)
        return self

    async def replace(self, source: Source) -> AsyncNewsroomRAG:
        async with self._lock:
            await asyncio.to_thread(self.rag.replace, source)
        return self

    async def search(self, query: str, *, top_k: int = 5, as_of: datetime | None = None,
                     before: datetime | None = None, after: datetime | None = None,
                     one_per_source: bool = False) -> tuple[Evidence, ...]:
        async with self._lock:
            return await asyncio.to_thread(self.rag.search, query, top_k=top_k, as_of=as_of,
                                           before=before, after=after,
                                           one_per_source=one_per_source)

    async def ask(self, question: str, *, top_k: int = 5, as_of: datetime | None = None,
                  before: datetime | None = None, after: datetime | None = None) -> Answer:
        async with self._lock:
            return await asyncio.to_thread(self.rag.ask, question, top_k=top_k,
                                           as_of=as_of, before=before, after=after)
