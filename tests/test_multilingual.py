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


class TestRoleAwareEmbeddings(unittest.TestCase):
    def test_explicit_prefixes_and_raw_evidence(self):
        calls = []
        class Model:
            def encode(self, texts, **kwargs):
                calls.append(texts)
                return [[1.0, 0.0] for _ in texts]
        adapter = MultilingualEmbedder(model=Model(), document_prefix='passage: ',
                                      query_prefix='query: ')
        rag = NewsroomRAG(embed=adapter, query_embed=adapter.embed_query,
                          lexical_weight=0).add(Source('a', 'عنوان', 'نص المصدر الأصلي'))
        result = rag.search('سؤال بالعربية')
        self.assertEqual(calls, [['passage: نص المصدر الأصلي'], ['query: سؤال بالعربية']])
        self.assertEqual(result[0].excerpt, 'نص المصدر الأصلي')
        rag.replace(Source('a', 'عنوان', 'نص محدث'))
        self.assertEqual(calls[-1], ['passage: نص محدث'])

    def test_query_hook_without_documents_rejected(self):
        from newsrag import ValidationError
        with self.assertRaises(ValidationError):
            NewsroomRAG(query_embed=lambda texts: [[1.0] for _ in texts])

    def test_empty_query_batch_no_download(self):
        adapter = MultilingualEmbedder(query_prefix='query: ')
        self.assertEqual(adapter.embed_query([]), [])
        self.assertIsNone(adapter._model)
