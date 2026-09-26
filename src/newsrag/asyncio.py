"""Async façade over the synchronous NewsroomRAG core.

The local search remains synchronous; provider calls run in worker threads. Do not
mutate a wrapped index directly while its façade has active operations.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from threading import Lock
from typing import TypeVar

from .core import Answer, Evidence, NewsroomRAG, Source

_T = TypeVar("_T")


class AsyncNewsroomRAG:
    """Awaitable wrapper for sync provider callables.

    A thread lock serializes work even if a caller cancels the awaiting task;
    cancellation does not stop an underlying provider call already in progress.
    """

    def __init__(self, rag: NewsroomRAG):
        self.rag = rag
        self._lock = Lock()

    def _call(self, operation: Callable[..., _T], *args: object,
              **kwargs: object) -> _T:
        with self._lock:
            return operation(*args, **kwargs)

    async def add(self, *sources: Source) -> AsyncNewsroomRAG:
        await asyncio.to_thread(self._call, self.rag.add, *sources)
        return self

    async def remove(self, source_id: str) -> AsyncNewsroomRAG:
        await asyncio.to_thread(self._call, self.rag.remove, source_id)
        return self

    async def replace(self, source: Source) -> AsyncNewsroomRAG:
        await asyncio.to_thread(self._call, self.rag.replace, source)
        return self

    async def search(self, query: str, *, top_k: int = 5, as_of: datetime | None = None,
                     before: datetime | None = None, after: datetime | None = None,
                     one_per_source: bool = False) -> tuple[Evidence, ...]:
        return await asyncio.to_thread(self._call, self.rag.search, query, top_k=top_k,
                                       as_of=as_of, before=before, after=after,
                                       one_per_source=one_per_source)

    async def ask(self, question: str, *, top_k: int = 5, as_of: datetime | None = None,
                  before: datetime | None = None, after: datetime | None = None) -> Answer:
        return await asyncio.to_thread(self._call, self.rag.ask, question, top_k=top_k,
                                       as_of=as_of, before=before, after=after)
