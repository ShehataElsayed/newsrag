import unittest

from newsrag import (
    NewsroomRAG,
    RetrievalCase,
    Source,
    ValidationError,
    evaluate_retrieval,
)


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.rag = NewsroomRAG().add(Source("a", "A", "alpha"),
                                     Source("b", "B", "beta"))

    def test_metrics_and_misses(self):
        cases = [RetrievalCase("alpha", frozenset({"a", "b"})),
                 RetrievalCase("absent", frozenset({"b"}))]
        score = evaluate_retrieval(self.rag, cases, top_k=1)
        self.assertEqual(score.recall_at_k, .25)
        self.assertEqual(score.hit_rate_at_k, .5)
        self.assertEqual(score.reciprocal_rank, .5)
        self.assertEqual(score.missed_queries, ("absent",))
        self.assertEqual(score.as_dict()["top_k"], 1)

    def test_empty_and_bad_cases(self):
        with self.assertRaises(ValidationError):
            evaluate_retrieval(self.rag, [])
        with self.assertRaises(ValidationError):
            RetrievalCase("", frozenset({"a"}))
        with self.assertRaises(ValidationError):
            RetrievalCase("alpha", frozenset())
        with self.assertRaises(ValidationError):
            evaluate_retrieval(self.rag, [RetrievalCase("alpha", frozenset({"a"}))], top_k=0)

    def test_unique_sources_and_rank(self):
        rag = NewsroomRAG(chunk_size=100, overlap=10).add(
            Source("a", "A", "alpha " * 100), Source("b", "B", "alpha"))
        score = evaluate_retrieval(rag, [RetrievalCase("alpha", frozenset({"b"}))], top_k=5)
        self.assertEqual(score.hit_rate_at_k, 1.0)
        self.assertGreater(score.reciprocal_rank, 0)
