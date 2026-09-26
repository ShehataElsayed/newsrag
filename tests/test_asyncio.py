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

    def test_cancelled_call_does_not_race_next_write(self):
        import threading
        started = threading.Event()
        release = threading.Event()
        def embed(texts):
            if texts[0] == "slow":
                started.set()
                release.wait(timeout=3)
            return [[1.0] for _ in texts]
        async def run():
            facade = AsyncNewsroomRAG(NewsroomRAG(embed=embed))
            task = asyncio.create_task(facade.add(Source("slow", "Slow", "slow")))
            await asyncio.to_thread(started.wait, 3)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            fast = asyncio.create_task(facade.add(Source("fast", "Fast", "fast")))
            await asyncio.sleep(.05)
            self.assertFalse(fast.done())
            release.set()
            await asyncio.wait_for(fast, 3)
            hits = await facade.search("slow fast", top_k=2)
            self.assertEqual({hit.source_id for hit in hits}, {"slow", "fast"})
        asyncio.run(run())
