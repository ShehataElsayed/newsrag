import unittest

from newsrag import NewsroomRAG, Source


class TestFusion(unittest.TestCase):
    def test_normalized_scores_do_not_let_bm25_swamp_semantics(self):
        vectors = {"alpha alpha alpha alpha": [0.0, 1.0], "other": [1.0, 0.0],
                   "alpha query": [1.0, 0.0]}
        rag = NewsroomRAG(embed=lambda texts: [vectors[text] for text in texts],
                          lexical_weight=0.2).add(
            Source("lex", "Lex", "alpha alpha alpha alpha"),
            Source("sem", "Sem", "other"))
        self.assertEqual(rag.search("alpha query")[0].source_id, "sem")

    def test_zero_weight_semantics_only(self):
        def embed(texts):
            return [[1.0, 0.0] for _ in texts]
        rag = NewsroomRAG(embed=embed, lexical_weight=0).add(Source("a", "A", "semantic"))
        self.assertEqual(rag.search("unrelated")[0].source_id, "a")
