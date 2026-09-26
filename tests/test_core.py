# ruff: noqa: DTZ001
import unittest
from datetime import datetime, timezone

from newsrag import NewsroomRAG, Source, ValidationError


class TestNewsroom(unittest.TestCase):
    def setUp(self):
        self.s = Source("a", "بيان", "أعلنت الشركة عن تجربة تجريبية ولم تعلن إطلاقا عاما.",
                        "https://example.org/a", datetime(2026, 9, 1, tzinfo=timezone.utc))

    def test_arabic_search_and_offsets(self):
        r = NewsroomRAG().add(self.s)
        e = r.search("اعلنت الشركه تجربة")[0]
        self.assertEqual(self.s.text[e.start:e.end], e.excerpt)
        self.assertEqual(e.ref, "E1")

    def test_no_match_abstains(self):
        a = NewsroomRAG().add(self.s).ask("penguin")
        self.assertEqual(a.evidence, ())
        self.assertIn("No matching evidence", a.text)

    def test_citation_checks(self):
        r = NewsroomRAG(generate=lambda prompt: "إطلاق [E1] و [E9]").add(self.s)
        a = r.ask("تجربة")
        self.assertFalse(a.citation_refs_valid)
        self.assertIn("Unknown citation IDs: E9", a.warnings)

    def test_date_filter_and_replace(self):
        r = NewsroomRAG().add(self.s)
        self.assertEqual(r.search("تجربة", after=datetime(2026, 9, 2, tzinfo=timezone.utc)), ())
        r.replace(Source("a", "New", "Completely different launch"))
        self.assertEqual(r.search("تجربة"), ())
        self.assertEqual(r.search("launch")[0].title, "New")

    def test_embedding_provider(self):
        def embed(texts):
            return [[1., 0.] for _ in texts]
        r = NewsroomRAG(embed=embed, lexical_weight=0).add(self.s)
        self.assertEqual(len(r.search("unrelated")), 1)

    def test_invalid_dates(self):
        with self.assertRaises(ValidationError):
            Source("a", "title", "text", published_at=datetime(2026, 1, 1))

    def test_duplicate(self):
        r = NewsroomRAG().add(self.s)
        with self.assertRaises(ValidationError):
            r.add(self.s)


if __name__ == "__main__":
    unittest.main()

class TestEdges(unittest.TestCase):
    def test_dates_zero_and_unknown(self):
        now = datetime(2026, 9, 26, tzinfo=timezone.utc)
        r = NewsroomRAG(clock=lambda: now, recency_half_life_days=30).add(
            Source("u", "Undated", "launch report"),
            Source("d", "Dated", "launch report", published_at=now),
        )
        self.assertEqual(r.search("launch", after=now)[0].source_id, "d")
        self.assertEqual(len(r.search("launch", after=now)), 1)
        self.assertEqual(r.search("launch")[0].source_id, "d")

    def test_chinese_and_arabic_no_diacritics(self):
        r = NewsroomRAG().add(Source("1", "x", "الْمُؤَسَّسَة أَعْلَنَتْ بِيَانًا"))
        self.assertTrue(r.search("المؤسسة أعلنت"))
        self.assertFalse(r.search("中文"))  # Basic lexical tokenizer isn't Chinese segmentation.

    def test_long_text_offsets_and_multiple_chunks(self):
        text = ("first second third " * 80).rstrip()
        r = NewsroomRAG(chunk_size=120, overlap=20).add(Source("a", "Long", text))
        self.assertGreater(len(r._chunks), 1)
        for c in r._chunks:
            self.assertEqual(text[c.start:c.end], c.text)
            self.assertLessEqual(len(c.text), 120)

    def test_one_per_source_and_stable_refs(self):
        r = NewsroomRAG(chunk_size=100, overlap=10).add(
            Source("a", "A", "alpha " * 100), Source("b", "B", "alpha"))
        e = r.search("alpha", top_k=5, one_per_source=True)
        self.assertEqual(len(e), 2)
        self.assertEqual([x.ref for x in e], ["E1", "E2"])

    def test_embed_failure_does_not_mutate(self):
        def embed(texts):
            return []
        r = NewsroomRAG(embed=embed)
        with self.assertRaises(ValidationError):
            r.add(Source("a", "A", "alpha"))
        self.assertEqual(r._sources, {})

    def test_invalid_provider_shape(self):
        r = NewsroomRAG(generate=lambda prompt: None).add(Source("a", "A", "alpha"))
        with self.assertRaises(ValidationError):
            r.ask("alpha")

class TestProviderValidation(unittest.TestCase):
    def test_nan_embedding_is_rejected_without_mutation(self):
        r = NewsroomRAG(embed=lambda texts: [[float("nan")] for _ in texts])
        with self.assertRaises(ValidationError):
            r.add(Source("a", "A", "alpha"))
        self.assertEqual(r._sources, {})

    def test_query_dimension_mismatch(self):
        def embed(texts):
            return [[1.0] if texts[0] == "alpha" else [1.0, 2.0]]
        r = NewsroomRAG(embed=embed).add(Source("a", "A", "alpha"))
        with self.assertRaises(ValidationError):
            r.search("beta")

class TestRerankAndNormalization(unittest.TestCase):
    def test_arabic_variants_match(self):
        corpus = Source("a", "Arabic", "إمرأة في المؤسسة", language="ar")
        rag = NewsroomRAG().add(corpus)
        self.assertTrue(rag.search("امرأه"))
        self.assertTrue(rag.search("الموسسه"))

    def test_rerank_references_follow_new_order(self):
        rag = NewsroomRAG(rerank=lambda query, hits: tuple(reversed(hits))).add(
            Source("a", "A", "alpha"), Source("b", "B", "alpha"))
        hits = rag.search("alpha", top_k=2)
        self.assertEqual([h.source_id for h in hits], ["b", "a"])
        self.assertEqual([h.ref for h in hits], ["E1", "E2"])

    def test_rerank_cannot_rewrite_or_inject(self):
        rag = NewsroomRAG(rerank=lambda query, hits: (Source("x", "X", "wrong"),)).add(
            Source("a", "A", "alpha"))
        with self.assertRaises(ValidationError):
            rag.search("alpha")

    def test_rerank_cannot_duplicate(self):
        rag = NewsroomRAG(rerank=lambda query, hits: (hits[0], hits[0])).add(
            Source("a", "A", "alpha"), Source("b", "B", "alpha"))
        with self.assertRaises(ValidationError):
            rag.search("alpha", top_k=2)

class TestInvalidNumerics(unittest.TestCase):
    def test_nonfinite_configuration(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValidationError):
                NewsroomRAG(lexical_weight=value)
            with self.assertRaises(ValidationError):
                NewsroomRAG(recency_half_life_days=value)

    def test_nonnumeric_embeddings(self):
        rag = NewsroomRAG(embed=lambda texts: [["not a float"] for _ in texts])
        with self.assertRaises(ValidationError):
            rag.add(Source("id", "Title", "sample"))
        self.assertEqual(rag._sources, {})

    def test_nonstr_source_values(self):
        with self.assertRaises(ValidationError):
            Source(1, "Title", "sample")
        with self.assertRaises(ValidationError):
            Source("id", "Title", "sample", published_at="yesterday")
