import unittest

from newsrag import MultilingualEmbedder, NewsroomRAG, Source


class FakeModel:
    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return [[1.0, 0.0] for _ in texts]


class TestMultilingualAdapter(unittest.TestCase):
    def test_model_injection_without_download(self):
        adapter = MultilingualEmbedder(model=FakeModel())
        self.assertEqual(adapter(["خبر", "news"]), [[1.0, 0.0], [1.0, 0.0]])
        rag = NewsroomRAG(embed=adapter, lexical_weight=0).add(Source("a", "A", "text"))
        self.assertEqual(rag.search("unrelated")[0].source_id, "a")

    def test_empty_batch_no_model_load(self):
        self.assertEqual(MultilingualEmbedder()([]), [])
