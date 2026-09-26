import asyncio
import unittest

from newsrag import AsyncNewsroomRAG, NewsroomRAG, Source


class TestAsyncFacade(unittest.TestCase):
    def test_add_search_ask_replace(self):
        async def run():
            rag = AsyncNewsroomRAG(NewsroomRAG(generate=lambda prompt: "Answer [E1]"))
            await rag.add(Source("a", "A", "alpha"))
            self.assertEqual((await rag.search("alpha"))[0].source_id, "a")
            self.assertTrue((await rag.ask("alpha")).citation_refs_valid)
            await rag.replace(Source("a", "A", "beta"))
            self.assertFalse(await rag.search("alpha"))
            self.assertTrue(await rag.search("beta"))
        asyncio.run(run())

    def test_concurrent_writes_are_serialized(self):
        async def run():
            rag = AsyncNewsroomRAG(NewsroomRAG())
            await asyncio.gather(*(rag.add(Source(str(i), str(i), "alpha")) for i in range(10)))
            self.assertEqual(len(await rag.search("alpha", top_k=10)), 10)
        asyncio.run(run())
